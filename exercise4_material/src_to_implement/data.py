import skimage.color
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # implement the Dataset class according to the description
    def __init__(self, data, mode):
        self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_tuple = []
        # get the item
        item = self.data.iloc[index]
        #print("item:", item)
        # separate treatment to the image
        img_color = gray2rgb(imread(item[0]))
        # print("ing:", img_color.shape)
        transform_img = self._transform(img_color).float()
        # punch two labels together
        labels = torch.tensor((item[1], item[2])).float()
        # append tensor data
        data_tuple.append(transform_img)
        data_tuple.append(labels)
        return data_tuple
