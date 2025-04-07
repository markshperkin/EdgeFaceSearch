import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchmetrics.detection import IntersectionOverUnion
from PIL import Image
import os
import cv2
import numpy as np
import json
import shutil

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

#Data base addresses
base_adress = 'Dataset_FDDB/Dataset_FDDB/images'
labels_adr = 'labels/label.txt'

# Make the labels ready

with open(labels_adr, 'r') as f:
    lines = f.readlines()
annotations = []
bboxes = []
flag = False
for line in lines:
    if line.startswith('#'):
      if flag:
        annotations.append({'image':img_name, 'bboxes': bboxes})
        bboxes = []
      flag = True
      img_name = line[2:]
    else:
      x_min, y_min, x_max, y_max = line.split()
      bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])

# Custom Dataset Class for FDDB
class FDDBDataset(Dataset):
    def __init__(self, img_dir, annot_file, target_size=(224, 224), transform=None):
        self.img_dir = img_dir
        self.target_size = target_size
        self.transform = transform
        self.data = self._parse_annotations(annot_file)

    def _parse_annotations(self, annot_file):
        
        data = []
        for el in annot_file:
            img_path = os.path.join(self.img_dir, el['image'][:-1])
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}. Skipping.")
                continue
            # Skip images with multiple bounding boxes
            if len(el['bboxes']) != 1:
                print(f"Multiple bounding boxes found for {img_path}. Skipping.")
                continue
            boxes = el['bboxes']
            data.append((img_path, boxes))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, boxes = self.data[idx]
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
        boxes_resized = []
        for box in boxes:
            x_min = int(box[0] * scale_x)
            y_min = int(box[1] * scale_y)
            x_max = int(box[2] * scale_x)
            y_max = int(box[3] * scale_y)
            boxes_resized.append([x_min, y_min, x_max, y_max])

        # Convert to tensor
        if self.transform:
            image_resized = self.transform(image_resized)
        else:
            image_resized = transforms.ToTensor()(image_resized)

        return image_resized, torch.tensor(boxes_resized, dtype=torch.float32)
    
# DataLoader preparation
def get_dataloaders(img_dir, annot_file, batch_size=16, target_size=(224, 224), validation_split=0.2):

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

def collate_fn(batch):
    """
    Custom collate function to handle variable-length bounding box arrays.

    :param batch: List of tuples (image, boxes).
    :return: Tuple of images and targets.
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

class MobileNetV2FaceDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2FaceDetector, self).__init__()
        # Load MobileNetV2 base
        self.base = models.mobilenet_v2(pretrained=pretrained).features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Custom head for bounding box and classification
        self.fc_bbox = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 4),  # Bounding box: [x_min, y_min, x_max, y_max]
        )
        self.fc_label = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 1),  # Binary classification: face/no face
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).view(x.size(0), -1)
        bbox = self.fc_bbox(x)
        label = self.fc_label(x)
        return bbox, label

model = MobileNetV2FaceDetector().to(device)

# Loss functions
bbox_loss_fn = nn.SmoothL1Loss()  # For bounding box regression
label_loss_fn = nn.BCELoss()      # For binary classification

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
def train_model(model, train_loader, val_loader, num_epochs=10):
    best_val_loss = float('inf')  # Initialize best validation loss
    best_model_path = "best_model.pth"  # Path to save the best model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            bboxes = [torch.tensor(t, dtype=torch.float32).to(device) for t in targets]  # List of bounding boxes
            labels = [int(1) for t in targets]  # List of labels
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            preds_bbox, preds_label = model(images)
            # Compute losses
            bbox_losses = []
            label_losses = []
            for i in range(len(bboxes)):
              bbox_losses.append(bbox_loss_fn(preds_bbox[i], bboxes[i]))
              label_losses.append(label_loss_fn(preds_label[i], labels[i].unsqueeze(-1)))

            bbox_loss = torch.mean(torch.stack(bbox_losses))
            label_loss = torch.mean(torch.stack(label_losses))
            loss = bbox_loss + label_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # Validate and save the best model
        val_loss = validate_model(model, val_loader)
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print("Training complete. Best model saved as:", best_model_path)

def normalize_boxes(preds):
    """
    Normalize the 'boxes' in the predictions to ensure they are all tensors of shape [N, 4].
    Args:
        preds: List of dictionaries with 'boxes' and 'labels'.
    Returns:
        Normalized predictions with 'boxes' as tensors of shape [N, 4].
    """
    for pred in preds:
        # If boxes is a list of tensors, stack them into a single tensor
        if isinstance(pred['boxes'], list):
            pred['boxes'] = torch.stack(pred['boxes'])  # Stack into [N, 4]
    return preds

def validate_model(model, val_loader):
    metric = IntersectionOverUnion().to(device)
    model.eval()
    total_bbox_loss = 0
    total_label_loss = 0
    total_iou = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            bboxes = [torch.tensor(t, dtype=torch.float32).to(device) for t in targets]  # List of bounding boxes
            labels = [int(1) for t in targets]  # List of labels
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            preds_bbox, preds_label = model(images)
            # print('labels')
            # print(preds_label)
            # input()
            bbox_losses = []
            label_losses = []
            # print(bboxes)
            # print('//////////////////////////////')
            for i in range(len(bboxes)):
              bbox_losses.append(bbox_loss_fn(preds_bbox[i], bboxes[i]))
              label_losses.append(label_loss_fn(preds_label[i], labels[i].unsqueeze(-1)))
            #   print([bboxes[i]])
            #   print('///////////////////////////////////////')
            #   print(preds_bbox[i].shape)
              preds = [
                {"boxes": [preds_bbox[i]], "labels": preds_label[i]}
                ]
              preds = normalize_boxes(preds)
            #   print("Preds")
            #   print(preds)
              targets_combined = torch.cat([bboxes[i]], dim=0)
                # print(targets_combined)
              targets = [
                {"boxes": targets_combined, "labels": torch.ones(len(targets_combined)).to(device)}
                ]
              iou_value = metric(preds, targets)
              total_iou.append(iou_value['iou'].item())
            #   targets = targets.to(device)
            #   print("Targets")
            #   print(targets)

            total_bbox_loss += torch.mean(torch.stack(bbox_losses))
            total_label_loss += torch.mean(torch.stack(label_losses))
            # print(targets)
            # loss = total_label_loss + total_label_loss

            # total_bbox_loss += bbox_loss_fn(preds_bbox, bboxes).item()
            # total_label_loss += label_loss_fn(preds_label, labels).item()

    # Calculate average validation loss
    avg_bbox_loss = total_bbox_loss / len(val_loader)
    avg_label_loss = total_label_loss / len(val_loader)
    val_loss = avg_bbox_loss + avg_label_loss
    print('IoU = ', sum(total_iou)/len(total_iou))
    print(f"Validation - BBox Loss: {avg_bbox_loss:.4f}, Label Loss: {avg_label_loss:.4f}, Total Loss: {val_loss:.4f}")
    return val_loss

batch_size = 32
target_size = (224, 224)
train_loader, val_loader = get_dataloaders(base_adress, annotations, batch_size, target_size)
train_model(model, train_loader, val_loader, num_epochs=100)

def calculate_iou(pred_box, gt_box):
    """
    Calculate IoU (Intersection over Union) for a single pair of boxes.
    Args:
        pred_box: Tensor of shape (4,), [x_min, y_min, x_max, y_max].
        gt_box: Tensor of shape (4,), [x_min, y_min, x_max, y_max].
    Returns:
        IoU value (float).
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    # Compute the area of intersection rectangle
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # Compute the area of both the predicted and ground-truth rectangles
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    # Compute the area of union
    union_area = pred_area + gt_area - inter_area

    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou
