
# Neural Architecture and Hyperparameter Search for Face Bounding Box Prediction on Edge

## Overview
In this project, the goal was to optimize a neural network architecture for face bounding box
prediction using a given facial image dataset. The primary objective was to design a lightweight
and efficient model capable of performing inference on resource-constrained hardware, NVIDIA
Jetson Nano. Additionally, a competition through a Kaggle evaluated the architecture based on
their prediction accuracy and inference latency. Accuracy was quantified using Intersection over
Union (IoU), a metric that assesses the degree of overlap between the model's predicted
bounding boxes and the ground-truth coordinates. Latency evaluation was performed directly
on the NVIDIA Jetson Nano to accurately reflect performance in a realistic deployment scenario
---
## Solution
In order to identify the optimal neural network architecture and corresponding
hyperparameters, I conducted a structured neural architecture and hyperparameter search.  
• The neural architecture search space included 21 different base CNN feature extractors.  
• The hyperparameter research considered learning rates ranging from 1e-2 to 1e-5.  
The network architecture employed a regression head consisting of two layers:  
• The input layer received neurons from the selected base architecture and reduced the
dimensionality by half  
• The output layer produced four values representing bounding box coordinates.  
The search procedure was structured in three distinct stages.  
➢ In the first stage, each combination of architecture and learning rate was trained and
evaluated for 10 epochs, after which I selected the top 8 architectures based on their
validation loss and inference latency. (There were 84 different configurations, and I was
not able to plot the results effectively)  
➢ In the second stage, these 8 selected architectures were further trained for 30 epochs
using their optimal learning rates identified from the initial stage. This stage narrowed
the selection down to the two best-performing architectures

![Validation Loss Graph](report/second_search_val_over_epoch.png)
![Latency Graph](report/second_search_latency_over_epoch.png)

Finally, the third stage involved extensive training of these two architectures over 60 epochs.
This allowed identification of the optimal architecture and learning rate for the bounding box
prediction task. 

![Validation Loss Graph](report/final_search_val_over_epoch.png)
![Latency Graph](report/final_search_latency_over_epoch.png)

The search concluded that the ResNet18 architecture with a learning rate of 0.01 was the optimal
architecture for this problem, achieving an inference latency of 0.013 seconds and an IoU
accuracy of 40%.
---
## how to run:
### Clone The Repo:
```bash
git clone https://github.com/markshperkin/EdgeFaceSearch
cd EdgeFaceSearch
```
### *OPTIONAL* install and use CUDA for Nvidia GPU
### Start the Search:  
*line 58 in secondSearch.py*: Modify the search space based on your requirement.  
*line 87 in secondSearch.py*: Modify the epoch number based on your requirement.   
*line 175 in secondSearch.py*: Modify the fitness function based on your requirement.
Note: Search may take a few hours depending on the GPU used, the size of the search space, and the number of epochs.
```bash
python .\secondSearch.py
```
### sort the results based on the selected fitness function:
```bash
python .\results.py
```
### plot the results:
```bash
python .\plot.py
```
### convert the trained model to .onnx:
```bash
python .\ONNX.py
```
### Train the optimal model:
Note: you need to implement the optimal model in lines 10-26 
```bash
python .\trainopt.pt
```
---
## Class Project

This project was developed as part of the Edge and Neuromorphic Computing class under the instruction of [Professor Ramtin Zand](https://sc.edu/study/colleges_schools/engineering_and_computing/faculty-staff/zand.php) and Teaching Assistants [Peyton Chandarana](https://www.peytonsc.com/) and [Mohammadreza Mohammadi](https://www.linkedin.com/in/mohammadreza-mohammadi-544837199/) with the caloboration of the [INTELLIGENT CIRCUITS, ARCHITECTURES, AND SYSTEMS LAB](https://www.icaslab.com/home) at the University of South Carolina.


