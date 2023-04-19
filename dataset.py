import os
import torch
import trimesh
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as dataset
from utils import get_json, get_medical_image, save_csv, get_csv, norm_zero_one


def convert(l):
    return list(map(float, l))


class VesselSegmentMeshDataSet(dataset):
    def __init__(self, json_path, indexes, isTrain=True, dire=''):
        super(VesselSegmentMeshDataSet, self).__init__()
        config = get_json(file=json_path)

        self.isTrain = isTrain
        self.images = []
        self.labels = []
        self.meshes = []

        for idx in indexes:
            items = config[idx]

            for item in items:
                image_path = item['image']
                label_path = item['label']
                meshs_path = item['verts']

                self.images.append(os.path.join(dire, image_path))
                self.labels.append(os.path.join(dire, label_path))
                self.meshes.append(os.path.join(dire, meshs_path))

    def __getitem__(self, index):


        image = sitk.ReadImage(self.images[index])
        verts = np.asarray(list(map(convert, get_csv(self.meshes[index], delimiter=' '))))
        verts = (verts - image.GetOrigin()) / image.GetSpacing()
        verts = 2. * verts / image.GetSize()[0] - 1.
        verts = verts[np.random.choice(len(verts), 3000)]

        image, _ = get_medical_image(image)
        image = norm_zero_one(image, span=[-200, 400])
        label, _ = get_medical_image(self.labels[index])

        return image[np.newaxis, :, :, :], label[np.newaxis, :, :, :], verts

    def __len__(self):
        return len(self.images)


def get_dataset():
    dire = './data/CoronaryArtery'
    json_path = './data.json'

    indexes = list(get_json(json_path).keys())
    train_index = indexes[:-2]
    valid_index = indexes[-2:]

    train_dataset = VesselSegmentMeshDataSet(json_path, train_index, isTrain=True, dire=dire)
    valid_dataset = VesselSegmentMeshDataSet(json_path, valid_index, isTrain=False, dire=dire)

    return train_dataset, valid_dataset


if __name__ == "__main__":
    train_dataset, valid_dataset = get_dataset()

    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    for image, label, mesh in dataloader:
        print(image.size())
        print(label.size())
        print(mesh.size())
        print(mesh.max())
        print(mesh.min())
        print('===========================================')

        exit()
