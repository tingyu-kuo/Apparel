import os
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PredictDataset(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.x = []
        self.n = [] # file name
        self.get_files()
        self.transform =  transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_files(self):
        files = os.listdir(self.path)
        for f_name in files:
            try:
                # 測試是否為可開啟的檔案格式
                Image.open(self.path / f_name)
                self.x.append(self.path / f_name)
                self.n.append(f_name)
            except:
                continue
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        image = self.transform(image)
        return image, self.n[index]
