import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dataloader import get_dataloaders, load_annotations

class RegressionHead(nn.Module):
    def __init__(self, in_features):
        super(RegressionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, 4),
        )
    
    def forward(self, x):
        return self.fc(x)

# complete model
class DetectionModel(nn.Module):
    def __init__(self, base, feature_dim):
        super(DetectionModel, self).__init__()
        self.base = base
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = RegressionHead(feature_dim)
    
    def forward(self, x):
        features = self.base(x)
        pooled = self.pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        bbox = self.head(flattened)
        return bbox

def compute_iou(box_pred, box_gt):
    x1 = max(box_pred[0], box_gt[0])
    y1 = max(box_pred[1], box_gt[1])
    x2 = min(box_pred[2], box_gt[2])
    y2 = min(box_pred[3], box_gt[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    union_area = area_pred + area_gt - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_base(arch_name):
    model_constructor = getattr(models, arch_name)
    model = model_constructor(pretrained=False)
    # Special handling for some models:
    if arch_name.startswith("alexnet") or arch_name.startswith("vgg") or arch_name.startswith("squeezenet"):
        base = model.features  # these have .features attribute
    else:
        base = nn.Sequential(*list(model.children())[:-1]) # get just the feature extractor (CNN), without the fully connected layers
    return base

search_space = [
    # ("alexnet", 256, 1e-05),
    # ("vgg11", 512, 0.01),
    ("vgg11_bn", 512, 0.0001), # final
    ("resnet18", 512, 0.001), # final winner
    # ("squeezenet1_0", 512, 1e-05),
    # ("squeezenet1_1", 512, 0.01),
    # ("densenet121", 1024),
    # ("shufflenet_v2_x0_5", 1024),
    # ("shufflenet_v2_x1_0", 1024),
    # ("mobilenet_v2", 1280, 0.001),
    # ("mobilenet_v3_small", 576),
    # ("mobilenet_v3_large", 960),
    # ("mnasnet0_5", 1280),
    # ("mnasnet1_0", 1280),
    # ("mnasnet1_3", 1280),
    # ("regnet_y_400mf", 440),
    # ("regnet_y_800mf", 784),
    # ("regnet_x_400mf", 400),
    # ("regnet_x_800mf", 672),
    # ("convnext_tiny", 768, 1e-05),
    # ("convnext_small", 768),
]

# learning rates to test
# learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
# learning_rates = [1e-4]


NUM_EPOCHS = 60
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_ADDRESS = "Dataset_FDDB/Dataset_FDDB/images"
LABELS_ADR = "labels/label.txt"

annotations = load_annotations(LABELS_ADR)
train_loader, val_loader = get_dataloaders(BASE_ADDRESS, annotations, batch_size=BATCH_SIZE, target_size=(224,224), validation_split=0.2)


# GPU warm up for inference evaluation
def warmup(model, device, iterations):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)

def validate_model(model, val_loader):
    bbox_loss_fn = nn.SmoothL1Loss()
    model.eval()
    total_loss = 0
    latency_list = []
    warmup(model, DEVICE, iterations = 2)

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            bboxes = [t.to(DEVICE) for t in targets]
            start = time.time()
            preds_bbox = model(images)
            end = time.time()
            latency_list.append(end - start)
            bbox_losses = []
            for i in range(len(bboxes)):
                bbox_losses.append(bbox_loss_fn(preds_bbox[i], bboxes[i]))
            
            loss = torch.mean(torch.stack(bbox_losses))
            total_loss += loss.item()
    
    avg_latency = sum(latency_list) / len(latency_list)
    avg_loss = total_loss / len(val_loader)
    return avg_loss, avg_latency

results = []

for arch_name, feature_dim, lr in search_space:
    print(f"Testing base: {arch_name} (feature dim = {feature_dim})")
    print(f"  Testing learning rate: {lr}")
    base = get_base(arch_name)
    model = DetectionModel(base, feature_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_avg_latencies = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(DEVICE)
            bboxes = [t.to(DEVICE) for t in targets]
            
            preds_bbox = model(images)
            
            bbox_losses = []
            for i in range(len(bboxes)):
                bbox_losses.append(criterion(preds_bbox[i], bboxes[i]))
            
            loss = torch.mean(torch.stack(bbox_losses))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_train_losses.append(total_loss / len(train_loader))
        
        val_loss, avg_latency = validate_model(model, val_loader)
        epoch_val_losses.append(val_loss)
        # epoch_avg_ious.append(avg_iou)
        epoch_avg_latencies.append(avg_latency)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {total_loss/len(train_loader):.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Avg Latency: {avg_latency:.6f} sec")
        
    fitness = epoch_val_losses[-1] * epoch_avg_latencies[-1]
    print(fitness)

    for epoch in range(NUM_EPOCHS):
        results.append({
            "architecture": arch_name,
            "feature_dim": feature_dim,
            "learning_rate": lr,
            "epoch": epoch + 1,
            "train_loss": epoch_train_losses[epoch],
            "val_loss": epoch_val_losses[epoch],
            "avg_latency": epoch_avg_latencies[epoch],
            "fitness": fitness
        })

csv_file = "second_search_results2.csv"
with open(csv_file, "w", newline="") as f:
    fieldnames = ["architecture", "feature_dim", "learning_rate", "epoch", "train_loss", "val_loss", "avg_latency", "fitness"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Search complete. Results saved to {csv_file}")

