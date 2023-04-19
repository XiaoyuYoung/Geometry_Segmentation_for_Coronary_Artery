from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_edge_loss, mesh_normal_consistency
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import torch.nn as nn


class MeshLoss(nn.Module):
    def __init__(self, weight_chamfer=1.0, num_samples=5000, weight_edge=1.0, weight_norm=0.1, weight_lapa=0.1):
        super(MeshLoss, self).__init__()
        self.weight_chamfer = weight_chamfer
        self.weight_edge = weight_edge
        self.weight_norm = weight_norm
        self.weight_lapa = weight_lapa

    def forward(self, input, target, *args):
        '''
        :param inputs: [verts, faces]
                    - in_verts (batch, num_point_{p}, 3)
                    - faces    (batch, num_faces_{p}, 3)
        :param target: ground truth verts (num_point_{t}, 3)
        :return:
        '''
        in_verts, faces = input
        gt_verts = target

        mesh = Meshes(verts=in_verts, faces=faces)

        chamfer = chamfer_distance(sample_points_from_meshes(mesh, num_samples=gt_verts.size()[1]), gt_verts)[0]
        # chamfer = chamfer_distance(mesh.verts_padded(), gt_verts)[0]
        chamfer = chamfer * self.weight_chamfer

        lapa = mesh_laplacian_smoothing(mesh) * self.weight_lapa
        edge = mesh_edge_loss(mesh) * self.weight_edge
        norm = mesh_normal_consistency(mesh) * self.weight_norm
        return chamfer, lapa, edge, norm


class GraphLoss(nn.Module):

    def __init__(self, weight_chamfer=1.0, weight_edge=0.1, weight_norm=0.01, weight_lapa=0.1):
        super(GraphLoss, self).__init__()
        self.criterion = MeshLoss(weight_chamfer, weight_edge, weight_norm, weight_lapa)

    def forward(self, inputs, target, *args):
        _, p_verts, p_faces = inputs
        _, verts = target

        chamfer_loss, edge_loss, norm_loss, lapa_loss = 0., 0., 0., 0.
        for vert, face in zip(p_verts, p_faces):
            chamfer, lapa, edge, norm = self.criterion([vert, face], verts)

            chamfer_loss += chamfer
            edge_loss += edge
            norm_loss += norm
            lapa_loss += lapa

        loss = chamfer_loss + lapa_loss + \
               edge_loss + norm_loss

        return loss


def create_loss(weight=None, reduction='mean'):
    return GraphLoss()
