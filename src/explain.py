import shap
import torch
from torchvision import models
from src.data import load_data

def explain_model(data_dir="data", model_path="models/leaf_health_model.pth", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _ = load_data(data_dir, batch_size)
    
    # Define the model architecture
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjust for binary classification
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()

    # Fix in-place ReLU issues by replacing all in-place ReLU layers
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    
    # Get a batch of data
    background = next(iter(train_loader))[0][:100].to(device)  # Use 100 samples as background for SHAP
    explainer = shap.DeepExplainer(model, background)

    # Use the first batch from the train loader for explanation
    sample_data, _ = next(iter(train_loader))
    sample_data = sample_data.to(device)
    
    # Generate SHAP values
    shap_values = explainer.shap_values(sample_data)
    
    # Visualize SHAP values
    shap.image_plot([shap_values[0]], sample_data.cpu().numpy())
