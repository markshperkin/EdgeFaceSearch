import torch
from trainopt import FaceDetector  # Replace with the actual module name if needed

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model (.pth)
PATH = 'best_modelOPT.pth'

# Create model instance and load weights
model = FaceDetector().to(device)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()  # Set the model to evaluation mode

# Prepare a dummy input tensor with the appropriate shape
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Export the model to ONNX format (this creates a new ONNX file)
torch.onnx.export(
    model,                    # The PyTorch model
    dummy_input,              # Example input tensor for tracing the model
    "modelOPT.onnx",             # Output ONNX file name
    export_params=True,       # Export the trained parameters
    opset_version=13,         # ONNX opset version
    do_constant_folding=True, # Optimize constant folding for better performance
    input_names=["input"],    # Name the input tensor
    output_names=["output"],  # Name the output tensor
    dynamic_axes=None         # Static axes; change if you need dynamic shapes
)

print("Model successfully exported to ONNX!")
