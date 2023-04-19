from utils import norm_zero_one, get_json, get_medical_image, save_csv, get_yaml, save_mps
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import sample_points_from_meshes
from train import Trainer
from tqdm import tqdm
import numpy as np
import trimesh
import torch
import os


def predict_mesh():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config/config-predict.yaml", help="config file",
                        required=True)

    config = get_yaml(parser.parse_args().config)

    trainer = Trainer(config=config)
    trainer.init_network()

    json = './test-data.json'
    src_dire = './data/'
    dst_dire = './predict'
    info = get_json(json)
    os.makedirs(dst_dire, exist_ok=True)

    points = []
    for idx, item in tqdm(enumerate(info[list(info.keys())[-2]]), total=len(info[list(info.keys())[-2]])):


        image, param = get_medical_image(os.path.join(src_dire, item['image']))
        image = norm_zero_one(image, span=[-200, 400])[np.newaxis, np.newaxis, :, :, :]
        image = torch.from_numpy(image)

        reconst, p_verts, p_faces = trainer.predict(image)

        verts = p_verts[-1].detach().cpu().numpy()[0]
        faces = p_faces[-1].detach().cpu().numpy()[0]

        p_verts = sample_points_from_meshes(Meshes(verts=p_verts[-1], faces=p_faces[-1]), num_samples=20)[
            0].detach().cpu().numpy()
        p_verts = p_verts * 8 + item['point']

        points.append(p_verts)
        verts = verts * 8 + item['point']
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export('./predict/right-predict-{}.stl'.format(idx))


    points = np.concatenate(points, axis=0)
    save_csv('./CoronaryArtery-pointcloud.xyz', points, delimiter=' ')
    save_mps(points, './CoronaryArtery-pointcloud.mps')


if __name__ == '__main__':
    predict_mesh()
