# ==========================================
# FILE: train_liveness.py
# DESCRIPTION: Training pipeline for Physical Liveness Detection using EfficientNet
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os

# 1. Hyperparameters & Configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR = "./dataset"
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 2  # 0: live, 1: spoof

# Setup device (CUDA for RTX GPU, fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 2. Data Pipeline (ETL for Computer Vision)
# We apply aggressive augmentations to make the model robust against lighting and angles
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Simulates different webcams
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet standards
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def create_dataloaders():
    """Creates PyTorch DataLoaders for batch processing."""
    image_datasets = {
        key: datasets.ImageFolder(os.path.join(DATA_DIR, key), value)
        for key, value in data_transforms.items()
    }
    
    dataloaders = {
        key: DataLoader(image_datasets[key], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        for key in data_transforms.keys()
    }
    return dataloaders, image_datasets['train'].classes

# 3. Model Architecture Construction
def build_model():
    """Loads a pretrained EfficientNet and modifies the classifier head."""
    # Load model with pre-trained ImageNet weights
    model = timm.create_model(MODEL_NAME, pretrained=True)
    
    # Replace the final fully connected layer to output exactly 2 classes
    num_in_features = model.get_classifier().in_features
    model.classifier = nn.Linear(num_in_features, NUM_CLASSES)
    
    return model.to(device)

# 4. The Training Loop
def train_model():
    dataloaders, class_names = create_dataloaders()
    model = build_model()
    
    # Loss function for classification
    criterion = nn.CrossEntropyLoss()
    # Optimizer defines how the neural network updates its weights
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"📦 Classes mapping: {class_names}")
    print("🔥 Starting training process...")

    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model if it's the best performing one
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'liveness_efficientnet.pth')
                print("💾 Saved new best model weights!")

    print(f"\n🎉 Training complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    # Ensure the dataset folder exists before running
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Dataset directory '{DATA_DIR}' not found.")
        print("Please create it and add 'live' and 'spoof' subfolders with images.")
    else:
        train_model()
