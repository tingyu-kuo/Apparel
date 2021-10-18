import os
import copy
import torch
import torch.nn as nn
from pathlib import Path
from dataset import PredictDataset
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader


class Predict():
    def __init__(self, in_path=os.getcwd(), out_path=None):
        self.path = Path(in_path)
        if out_path:
            self.save = Path(out_path)
        else:
            self.save = self.path
        self.get_path()
        self.get_device()
        self.load_model()
        self.load_data()
        self.get_classes()

    def get_path(self):
        # self.base_path = Path('./', 'lib', 'models')
        self.type_model_path = str('0225-resnext101-type.pth')        # acc = 96%
        self.product_model_path = str('0225-resnext101-product.pth')    # acc = 95.5%
        self.color_model_path = str('0223-resnext101-color.pth')       # acc = 95%

    def get_device(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')

    def load_model(self):
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=False)
        # type predict model
        self.type_model = copy.deepcopy(model)
        self.type_model.fc = nn.Linear(self.type_model.fc.in_features, 2)
        self.type_model.load_state_dict(torch.load(self.type_model_path, map_location=self.device))
        if self.use_cuda:
            self.type_model.cuda(self.device)
        self.type_model.eval()
        # product predict model
        self.product_model = copy.deepcopy(model)
        self.product_model.fc = nn.Linear(self.product_model.fc.in_features, 6)
        self.product_model.load_state_dict(torch.load(self.product_model_path, map_location=self.device))
        if self.use_cuda:
            self.product_model.cuda(self.device)
        self.product_model.eval()
        # color predict model
        self.color_model = copy.deepcopy(model)
        self.color_model.fc = nn.Linear(self.color_model.fc.in_features, 12)
        self.color_model.load_state_dict(torch.load(self.color_model_path, map_location=self.device))
        if self.use_cuda:
            self.color_model.cuda(self.device)
        self.color_model.eval()

    def load_data(self):
        self.dataset = PredictDataset(self.path)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=2, num_workers=4)

    def get_classes(self):
        self.type_classes = ['2', '1']      # 平織1，針織2
        self.product_classes = ['JK', 'SK', 'SH', 'DS', 'PT', 'TP']
        self.color_classes = ['WH', 'OG', 'PK', 'GY', 'BK', 'PU', 'BU', 'GN', 'UN', 'BN', 'YE', 'RD']
if __name__ == '__main__':
    Predict(in_path='C:\\Tingyu\\Study\\MVL\\clothes\\AI Import\\20201204\\Image_20201204\\19S001')


