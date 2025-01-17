import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.data import load_data

import torch.nn.init as init

def initialize_xavier(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

def train_model(data_dir="data", epochs=20, lr=0.001, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = load_data(data_dir, batch_size)
    
    # Initialize GoogLeNet
    model = models.googlenet(init_weights=True, aux_logits=True)  # Enable auxiliary outputs
    model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust for binary classification
    model = model.to(device)
    
    # Apply Xavier initialization to the model
    initialize_xavier(model)
    print("Xavier initialization applied to the model.")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            if isinstance(outputs, tuple):  # GoogLeNetOutputs
                main_output, aux_output = outputs[0], outputs[1]
                loss1 = criterion(main_output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2  # Weighted sum of main and auxiliary losses
            else:
                loss = criterion(outputs, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "models/leaf_health_model.pth")
    print("Modelo salvo em models/leaf_health_model.pth")
