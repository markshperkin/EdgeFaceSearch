import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def load_annotations(labels_adr):
    with open(labels_adr, 'r') as f:
        lines = f.readlines()
    annotations = []
    bboxes = []
    flag = False
    for line in lines:
        if line.startswith('#'):
            if flag:
                annotations.append({'image': img_name.strip(), 'bboxes': bboxes})
                bboxes = []
            flag = True
            img_name = line[2:].strip()
        else:
            x_min, y_min, x_max, y_max = line.split()
            bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    if flag:
        annotations.append({'image': img_name.strip(), 'bboxes': bboxes})
    return annotations

class FDDBDataset(Dataset):
    def __init__(self, img_dir, annot_file, target_size=(224, 224), transform=None):
        self.img_dir = img_dir
        self.target_size = target_size
        self.transform = transform
        self.data = self._parse_annotations(annot_file)
    
    def _parse_annotations(self, annot_file):
        data = []
        for el in annot_file:
            img_path = os.path.join(self.img_dir, el['image'])
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}. Skipping.")
                continue
            # use images with a single bounding box.
            if len(el['bboxes']) != 1:
                print(f"Multiple bounding boxes found for {img_path}. Skipping.")
                continue
            data.append((img_path, el['bboxes'][0]))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, box = self.data[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        # Original dimensions
        h, w, _ = image.shape
        # Resize image
        image_resized = cv2.resize(image, self.target_size)
        target_h, target_w = self.target_size
        # Scale bounding boxes
        scale_x = target_w / w
        scale_y = target_h / h
        x_min = int(box[0] * scale_x)
        y_min = int(box[1] * scale_y)
        x_max = int(box[2] * scale_x)
        y_max = int(box[3] * scale_y)
        box_resized = [x_min, y_min, x_max, y_max]
        # Convert to tensor
        if self.transform:
            image_resized = self.transform(image_resized)
        else:
            image_resized = transforms.ToTensor()(image_resized)
        return image_resized, torch.tensor(box_resized, dtype=torch.float32)

def collate_fn(batch):
    """
    Custom collate function to handle variable-length bounding box arrays.

    :param batch: List of tuples (image, boxes).
    :return: Tuple of images and targets.
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

def get_dataloaders(img_dir, annot_file, batch_size=16, target_size=(224, 224), validation_split=0.2):

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # Dataset
    dataset = FDDBDataset(img_dir, annot_file, target_size, transform)
    # Split dataset
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader
