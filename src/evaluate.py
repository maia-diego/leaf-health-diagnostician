import torch
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
from src.data import load_data

def evaluate_model(data_dir="data", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = load_data(data_dir, batch_size)

    # Define the model architecture
    model = models.googlenet()  # Set pretrained to False
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjust for binary classification
    model.load_state_dict(torch.load("models/leaf_health_model.pth", map_location=device))  # Load the state_dict
    model = model.to(device)
    model.eval()

    # Evaluate the model
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=["Healthy", "Diseased"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
