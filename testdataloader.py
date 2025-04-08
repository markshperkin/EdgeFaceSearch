
from dataloader import get_dataloaders, load_annotations

def main():
    base_address = 'Dataset_FDDB/Dataset_FDDB/images'
    labels_adr = 'labels/label.txt'
    
    annotations = load_annotations(labels_adr)
    
    # parameters for the dataloaders
    batch_size = 16
    target_size = (224, 224)
    
    train_loader, val_loader = get_dataloaders(base_address, annotations, batch_size, target_size)
    
    print(f"Number of training images: {len(train_loader.dataset)}")
    print(f"Number of validation images: {len(val_loader.dataset)}")

if __name__ == "__main__":
    main()
