# testdataloader.py

from dataloader import get_dataloaders, load_annotations

def main():
    # Define paths for your images and labels
    base_address = 'Dataset_FDDB/Dataset_FDDB/images'
    labels_adr = 'labels/label.txt'
    
    # Load annotations from the label file
    annotations = load_annotations(labels_adr)
    
    # Set parameters for the dataloaders
    batch_size = 16
    target_size = (224, 224)
    
    # Get train and validation dataloaders
    train_loader, val_loader = get_dataloaders(base_address, annotations, batch_size, target_size)
    
    # Print the number of images in each dataset
    print(f"Number of training images: {len(train_loader.dataset)}")
    print(f"Number of validation images: {len(val_loader.dataset)}")

if __name__ == "__main__":
    main()
