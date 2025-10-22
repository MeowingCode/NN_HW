from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.detection import SSD300_VGG16_Weights
import random
import json

class DetectionDataset(Dataset):
    def __init__(self, data_dict_file, transform=None, augment_bbox=None):
        """
        transform: простые аугментации, не трогающие боксы
        augment_bbox: сложные аугментации, изменяющие боксы (affine, scale, rotation)
        """
        self.transform = transform
        self.augment_bbox = augment_bbox

        # обязательные трансформации SSD
        self.ssd_transform = SSD300_VGG16_Weights.DEFAULT.transforms()

        self.data_dict_file = Path(data_dict_file)
        self.root_dir = self.data_dict_file.parent

        with open(data_dict_file, 'r') as f:
            self.data_dict = json.load(f)
        self.imgs = list(self.data_dict.keys())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(self.root_dir / img_path).convert("RGB")
        w, h = img.size

        # Bounding boxes
        boxes = []
        for bbox in self.data_dict[img_path]:
            xmin = bbox[0] * w
            xmax = bbox[1] * w
            ymin = bbox[2] * h
            ymax = bbox[3] * h
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        # --- сложные аугментации, изменяющие боксы ---
        if self.augment_bbox:
            img, boxes = self.augment_bbox(img, boxes)
            target["boxes"] = boxes

        # --- простые аугментации ---
        if self.transform:
            img = self.transform(img)

        # --- обязательные SSD трансформации ---
        img = self.ssd_transform(img)

        return img, target