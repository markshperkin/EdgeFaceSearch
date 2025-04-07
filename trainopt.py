import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataloader import get_dataloaders, load_annotations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceDetector(nn.Module):
    def __init__(self, pretrained=False):
        super(FaceDetector, self).__init__()
        # Load the base architecture
        self.base = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Head for bounding box regression
        self.fc_bbox = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
    
    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).view(x.size(0), -1)
        bbox = self.fc_bbox(x)
        return bbox

def train_model(model, train_loader, val_loader, num_epochs):
    bbox_loss_fn = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    best_model_path = "best_modelOPT.pth"
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            bboxes = [t.to(device) for t in targets]
            
            preds_bbox = model(images)
            
            bbox_losses = []
            for i in range(len(bboxes)):
                bbox_losses.append(bbox_loss_fn(preds_bbox[i], bboxes[i]))
            
            loss = torch.mean(torch.stack(bbox_losses))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        val_loss = validate_model(model, val_loader)
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
    
    print("Training complete. Best model saved as:", best_model_path)

def validate_model(model, val_loader):
    bbox_loss_fn = nn.SmoothL1Loss()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            bboxes = [t.to(device) for t in targets]
            preds_bbox = model(images)
            
            bbox_losses = []
            for i in range(len(bboxes)):
                bbox_losses.append(bbox_loss_fn(preds_bbox[i], bboxes[i]))
            
            loss = torch.mean(torch.stack(bbox_losses))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    base_address = 'Dataset_FDDB/Dataset_FDDB/images'
    labels_adr = 'labels/label.txt'
    annotations = load_annotations(labels_adr)
    batch_size = 32
    target_size = (224, 224)
    
    train_loader, val_loader = get_dataloaders(base_address, annotations, batch_size, target_size)
    model = FaceDetector().to(device)
    train_model(model, train_loader, val_loader, num_epochs=100)

if __name__ == "__main__":
    main()
