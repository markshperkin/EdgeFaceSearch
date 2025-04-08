import os
import cv2
import torch
import time
import csv
from torchvision import transforms
from trainopt import FaceDetector  # Import your model definition

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model = FaceDetector().to(device)
    best_model_path = "best_modelOPT.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_dir = "test"
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg'))]
    
    results = []
    total_time = 0.0
    num_images = 0
    
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            continue

        # original image dimensions
        orig_h, orig_w, _ = img.shape
        # print(orig_w)
        # print(img.shape)

        img_resized = cv2.resize(img, (224, 224))

        input_tensor = transform(img_resized)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            bbox_pred = model(input_tensor)
        end_time = time.time()

        inference_time = end_time - start_time
        total_time += inference_time
        num_images += 1

        bbox_pred = bbox_pred.squeeze(0).detach().cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox_pred)
        # print("before: ")
        # print(x1,y1,x2,y2)
        
        # convert predictions to original image dimensions
        x1 = int(x1 * (orig_w / 224))
        y1 = int(y1 * (orig_h / 224))
        x2 = int(x2 * (orig_w / 224))
        y2 = int(y2 * (orig_h / 224))
        # print("after: ")
        # print(x1,y1,x2,y2)

        results.append([f"'{image_file}'", x1, y1, x2, y2])

    
    avg_inference_time = total_time / num_images if num_images > 0 else 0.0
    print(num_images)
    print(f"Average inference time per image: {avg_inference_time:.4f} seconds")
    
    csv_file = "results-opt2.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "x1", "y1", "x2", "y2"])
        writer.writerows(results)
    
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    main()
