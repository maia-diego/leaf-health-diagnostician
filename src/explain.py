import shap
import torch
from src.data import load_data

def explain_model(data_dir="data", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _ = load_data(data_dir, batch_size)
    
    model = torch.load("models/leaf_health_model.pth", map_location=device)
    model.eval()
    
    background = next(iter(train_loader))[0][:100].to(device)
    explainer = shap.DeepExplainer(model, background)
    
    sample_data, _ = next(iter(train_loader))
    sample_data = sample_data.to(device)
    shap_values = explainer.shap_values(sample_data)
    
    shap.image_plot([shap_values[0]], sample_data.cpu().numpy())
