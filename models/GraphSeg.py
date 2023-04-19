import torch
import trimesh
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GCNConv


class Sphere(object):
    '''
    load three spheres (interpolation)
    '''

    def __init__(self):
        '''
        Initialize the sphere
        verts [num_verts, 3]
        edges [2, num_edges] : [torch.Size([2, 480]), torch.Size([2, 1920]), torch.Size([2, 7680])]
        faces [num_faces, 3] : [torch.Size([320, 3]), torch.Size([1280, 3]), torch.Size([5120, 3])]
        '''
        self.spheres = [trimesh.load('./utils/sphere-{}.obj'.format(i)) for i in range(3)]
        self.verts = [torch.from_numpy(np.asarray(mesh.vertices).copy()).float() for mesh in self.spheres]
        self.edges = [torch.from_numpy(np.asarray(mesh.edges_unique).copy()).long().transpose(1, 0) for mesh in
                      self.spheres]
        self.faces = [torch.from_numpy(np.asarray(mesh.faces).copy()).long() for mesh in self.spheres]


class GResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GResBlock, self).__init__()
        self.model = Sequential('x, edge_index', [
            (GCNConv(in_channels=in_dim, out_channels=hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(in_channels=hidden_dim, out_channels=in_dim), 'x, edge_index -> x'),
            nn.ReLU()
        ])

    def forward(self, x, edge_index):
        '''
        :param x: [batch, num_verts, channel]
        :param edge_index: [2, num_edges]
        :return:
        '''
        return (x + self.model(x, edge_index)) * 0.5


class GBottleNeck(nn.Module):
    def __init__(self, block_num, in_dim, hidden_dim, out_dim):
        super(GBottleNeck, self).__init__()

        self.model = Sequential('x, edge_index', [
            (GCNConv(in_channels=in_dim, out_channels=hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            *[(GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim), 'x, edge_index -> x') for _ in range(block_num)]
        ])
        self.gconv = GCNConv(in_channels=hidden_dim, out_channels=out_dim)

    def forward(self, x, edge_index):
        '''
        :param x: [batch, num_verts, channel]
        :param edge_index: [2, num_edges]
        :return:
        '''
        x_hidden = self.model(x=x, edge_index=edge_index)
        return self.gconv(x_hidden, edge_index), x_hidden


class GProjection3D(nn.Module):
    ## https://github.com/cvlab-epfl/voxel2mesh
    def __init__(self, num_feats):
        super(GProjection3D, self).__init__()

        self.sum_neighbourhood = nn.Conv2d(num_feats, num_feats, kernel_size=(1, 27), padding=0)

        self.shift_delta = nn.Conv1d(num_feats, 27 * 3, kernel_size=(1), padding=0)
        self.shift_delta.weight.data.fill_(0.0)
        self.shift_delta.bias.data.fill_(0.0)

        self.feature_diff_1 = nn.Linear(num_feats + 3, num_feats)
        self.feature_diff_2 = nn.Linear(num_feats, num_feats)

        self.feature_center_1 = nn.Linear(num_feats + 3, num_feats)
        self.feature_center_2 = nn.Linear(num_feats, num_feats)

    def forward(self, images, verts):
        B, N, _ = verts.shape

        w, h, d = verts[:, :, 0], verts[:, :, 1], verts[:, :, 2]

        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)
        d = torch.clamp(d, min=-1, max=1)

        verts = torch.stack([w, h, d], dim=-1)

        center = verts[:, :, None, None]
        features = F.grid_sample(images, center, mode='bilinear', padding_mode='border', align_corners=True)
        features = features[:, :, :, 0, 0]
        shift_delta = self.shift_delta(features).permute(0, 2, 1).view(B, N, 27, 1, 3)
        shift_delta[:, :, 0, :, :] = shift_delta[:, :, 0, :, :] * 0

        neighbourhood = verts[:, :, None, None] + shift_delta
        features = F.grid_sample(images, neighbourhood, mode='bilinear', padding_mode='border',
                                 align_corners=True)
        features = features[:, :, :, :, 0]
        features = torch.cat([features, neighbourhood.permute(0, 4, 1, 2, 3)[:, :, :, :, 0]], dim=1)

        features_diff_from_center = features - features[:, :, :, 0][:, :, :,
                                               None]  # 0 is the index of the center cordinate in shifts
        features_diff_from_center = features_diff_from_center.permute([0, 3, 2, 1])
        features_diff_from_center = self.feature_diff_1(features_diff_from_center)
        features_diff_from_center = self.feature_diff_2(features_diff_from_center)
        features_diff_from_center = features_diff_from_center.permute([0, 3, 2, 1])

        features_diff_from_center = self.sum_neighbourhood(features_diff_from_center)[:, :, :, 0].transpose(2, 1)

        center_feautres = features[:, :, :, 13].transpose(2, 1)
        center_feautres = self.feature_center_1(center_feautres)
        center_feautres = self.feature_center_2(center_feautres)

        features = center_feautres + features_diff_from_center
        return features


class GUnpooling(nn.Module):
    '''
    Unpooling verts in graph
    '''

    def __init__(self):
        super(GUnpooling, self).__init__()

    def forward(self, x, edge_index):
        new_features = x[:, edge_index.transpose(1, 0)].clone()
        new_vertices = 0.5 * new_features.sum(2)
        output = torch.cat([x, new_vertices], 1)

        return output


class GraphSeg(nn.Module):
    def __init__(self, coords_dim=3, hidden_dim=192, num_blcoks=6, feats_dims=[128, 64, 32]):
        super(GraphSeg, self).__init__()

        self.init_sphere = Sphere()

        ## network parameter

        ## network
        self.gcn_projn = nn.ModuleList([
            GProjection3D(num_feats) for num_feats in feats_dims
        ])
        self.gcn_model = nn.ModuleList([
            GBottleNeck(block_num=num_blcoks, in_dim=sum(feats_dims) + coords_dim,
                        hidden_dim=hidden_dim, out_dim=coords_dim),
            GBottleNeck(block_num=num_blcoks, in_dim=sum(feats_dims) + hidden_dim + coords_dim,
                        hidden_dim=hidden_dim, out_dim=coords_dim),
            GBottleNeck(block_num=num_blcoks, in_dim=sum(feats_dims) + hidden_dim + coords_dim,
                        hidden_dim=hidden_dim, out_dim=hidden_dim),
        ])
        self.gcn_final = Sequential(
            'x, edge_index', [
                (nn.ReLU(), 'x -> x'),
                (GCNConv(in_channels=hidden_dim, out_channels=coords_dim), 'x, edge_index -> x')
            ]
        )
        self.gcn_unpol = GUnpooling()

    def forward(self, img_feats):
        '''
        :param img_feats: [torch.Size([batch, 128, 8, 8,  8 ]),
                           torch.Size([batch, 64, 16, 16, 16]),
                           torch.Size([batch, 32, 32, 32, 32])]
        :return: verts  x1-[batch, 162, 3]  x2-[batch, 642,  3]  x3-[batch, 2562, 3]
                 faces     [batch, 320, 3]     [batch, 1280, 3]     [batch, 5120, 3]
        '''

        batch = img_feats[0].size()[0]
        device = img_feats[0].device

        ## Initilization
        ## init_verts [batch, 162, 3]
        ## init_edges [torch.Size([2, 480]), torch.Size([2, 1920]), torch.Size([2, 7680])]
        init_verts = self.init_sphere.verts[0].unsqueeze(dim=0).expand(batch, -1, -1).to(device)
        init_edges = [e.to(device) for e in self.init_sphere.edges]
        init_faces = [f.unsqueeze(dim=0).expand(batch, -1, -1).to(device) for f in self.init_sphere.faces]

        ## GCN Block #1
        ## x1_proj   [batch, 162, num_feats]
        ## x1        [batch, 162，coord_dim]
        ## x_hidden  [batch, 162, hidden_dim]
        x1_proj = torch.cat([self.gcn_projn[idx](images=feats, verts=init_verts)
                             for idx, feats in enumerate(img_feats)], dim=2)
        x1, x_hidden = self.gcn_model[0](x=torch.cat([x1_proj, init_verts], dim=2), edge_index=init_edges[0])
        x1 = x1 + init_verts

        # ===============================================================

        ## GCN Block #2
        ## x1_unpool        [batch, 642, coord_dim]
        ## x1_hidden_unpool [batch, 642, hidden_dim]
        ##
        ## x2_proj          [batch, 642, num_feats]
        ## x2               [batch, 642, coord_dim]
        ## x_hidden         [batch, 642, hidden_dim]
        x1_unpool = self.gcn_unpol(x=x1, edge_index=init_edges[0])
        x1_hidden_unpool = self.gcn_unpol(x=x_hidden, edge_index=init_edges[0])

        x2_proj = torch.cat([self.gcn_projn[idx](images=feats, verts=x1_unpool)
                             for idx, feats in enumerate(img_feats)], dim=2)
        x2, x_hidden = self.gcn_model[1](
            x=torch.cat([x1_hidden_unpool, x2_proj, x1_unpool], dim=2), edge_index=init_edges[1])
        x2 = x2 + x1_unpool

        # ===============================================================

        ## GCN Block #3
        ## x2_unpool        [batch, 2562, coord_dim]
        ## x2_hidden_unpool [batch, 2562, hidden_num]
        ##
        ## x3_proj          [batch, 2562, num_feats]
        ## x3               [batch, 2562, hidden_num]
        x2_unpool = self.gcn_unpol(x=x2, edge_index=init_edges[1])
        x2_hidden_unpool = self.gcn_unpol(x=x_hidden, edge_index=init_edges[1])

        x3_proj = torch.cat([self.gcn_projn[idx](images=feats, verts=x2_unpool)
                             for idx, feats in enumerate(img_feats)], dim=2)
        x3, _ = self.gcn_model[2](
            x=torch.cat([x2_hidden_unpool, x3_proj, x2_unpool], dim=2), edge_index=init_edges[2])

        # ===============================================================

        ## GCN Block FINAL
        x3 = self.gcn_final(x3, init_edges[2])
        x3 = x3 + x2_unpool

        return [x1, x2, x3], init_faces


def create_model(coords_dim=3, hidden_dim=192, feats_dims=[128, 64, 32], *args):
    gcn = GraphSeg(coords_dim=coords_dim, hidden_dim=hidden_dim, feats_dims=feats_dims)
    return gcn
