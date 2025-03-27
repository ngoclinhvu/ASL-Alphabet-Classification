import numpy as np
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

import torchvision
import torchvision.transforms as transform

from efficientnet_pytorch import EfficientNet

# pip install efficientnet_pytorch
# Each image hasn't had label yet, how to add y_train
# test_dataset don't have delete_test image, how to modify CustomDataset class so that class_idx won't go wrong

class CustomDataset(Dataset):
    def __init__(self, root_dir, dict, num_classes, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Walk through all sub-folders
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    # Find image path
                    img_path = os.path.join(class_dir, img_name)
                    # Find label
                    class_idx = dict[class_name]
                    # Create class tensor
                    class_tensor = torch.zeros(num_classes, dtype = torch.float32)
                    class_tensor[class_idx] = 1
                    
                    # Append image path and label to the list
                    self.image_paths.append(img_path)
                    self.labels.append(class_tensor)
            
            else:
                if class_dir:
                    # Find image path
                    img_path = class_dir
                    # Find label
                    class_idx = dict[class_name.split('_')[0]]
                    # Create class tensor
                    class_tensor = torch.zeros(num_classes, dtype = torch.float32)
                    class_tensor[class_idx] = 1
                    
                    # Append image path and label to the list
                    self.image_paths.append(img_path)
                    self.labels.append(class_tensor)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        
        # Create EfficientNet model
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        # Freeze all parameters of the model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Modify the last linear layer
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
        
        # Only train the last 5 layers
        for param in self.model._blocks[-5:].parameters():
            param.requires_grad = True
            
        # Add Sigmoid function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x

# Preprocess images
transform = transform.Compose([
    transform.Resize((224,224)),
    transform.RandomRotation(15),
    transform.ToTensor(),
    transform.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Number of classes and batch size
    num_classes = 29
    batch_size = 128

    # Dictionary to translate class names to indexes
    class_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                    'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                    'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}

    # Load input
    train_dataset = CustomDataset(root_dir = '/home/ducanh/NgocLinh/ASLAlphabetClassification/data/asl_alphabet_train/asl_alphabet_train'
                                  , dict=class_to_index, num_classes=num_classes, transform = transform)
    test_dataset = CustomDataset(root_dir = '/home/ducanh/NgocLinh/ASLAlphabetClassification/data/asl_alphabet_test/asl_alphabet_test'
                                 , dict=class_to_index, num_classes=num_classes, transform = transform)

    # Pass input into dataloader
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    # Initialize model
    model = CustomEfficientNet(num_classes)
    
    # Move model to GPU
    model = model.to(device)

    # Define loss function, optimizer, and scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model._blocks[-5:].parameters(), lr = 0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    
    # Training loop
    num_epochs = 30
    train_loss = 0.0
    lambda_reg = 0.1
    for epoch in range(num_epochs):
        model.train()
        correct_train_predictions = 0

        for X_train, y_train in tqdm(train_dataloader):
            # Move train data to GPU
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            y_train_predicted = model(X_train)
            loss = loss_fn(y_train_predicted, y_train)
            
            # Calculate L2 Regularization
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            train_loss += loss * y_train.size(0) + lambda_reg * l2_norm

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate number of correct predictions
            label_train = torch.argmax(y_train, dim=1)
            label_train_predicted = torch.argmax(y_train_predicted, dim=1)
            correct_train_predictions += (label_train_predicted == label_train).sum().item()
            
        # Calculate train accuracy and train loss
        train_accuracy = correct_train_predictions / len(train_dataloader.dataset)
        train_loss /= len(train_dataloader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}")
        
        # Implement Learning Rate Scheduler
        scheduler.step()

    model.eval()
    correct_test_predictions = 0
    with torch.no_grad():
        for X_test, y_test in tqdm(test_dataloader):
            # Move test data to GPU
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            
            # Predict label and calculate accuracy
            y_test_predicted = model(X_test)
            
            # Calculate test accuracy
            label_test = torch.argmax(y_test, dim=1)
            label_test_predicted = torch.argmax(y_test_predicted, dim=1)
            correct_test_predictions += (label_test_predicted == label_test).sum().item()
        
        test_accuracy = correct_test_predictions/len(test_dataloader.dataset)
        print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()


# Convert index to class name
# idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}  # Reverse mapping
# predicted_class = idx_to_class[predicted_label]  # Convert label to class name