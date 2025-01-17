import logging
import torch
from torchvision import models, transforms
from PIL import Image
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def classify_images(image_dir, model_path="models/leaf_health_model.pth", output_file="classification_results.txt"):
    # Device setup
    logging.info("Setting up device for inference.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the GoogLeNet model
    logging.info(f"Loading model from {model_path}.")
    model = models.googlenet(init_weights=False)  # Disable auxiliary outputs for inference
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjust for binary classification
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logging.info("Model loaded successfully and set to evaluation mode.")

    # Define the preprocessing pipeline
    logging.info("Defining preprocessing transformations.")
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Class labels
    class_names = ["Diseased", "Healthy"]
    logging.info(f"Class labels: {class_names}")

    # Check the image directory
    if not os.path.exists(image_dir):
        logging.error(f"Image directory {image_dir} does not exist.")
        return
    logging.info(f"Processing images from directory: {image_dir}")

    # Iterate through the images in the directory
    results = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, filename)
            logging.info(f"Processing image: {image_path}")
            try:
                # Open and preprocess the image
                image = Image.open(image_path).convert("RGB")
                input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

                # Perform inference
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted_class = torch.max(output, 1)
                
                # Append the result
                result = f"{filename}: {class_names[predicted_class.item()]}"
                results.append(result)
                logging.info(f"Prediction for {filename}: {class_names[predicted_class.item()]}")
            except Exception as e:
                logging.error(f"Error processing image {filename}: {e}")

    # Save results to a file
    try:
        with open(output_file, "w") as f:
            f.write("\n".join(results))
        logging.info(f"Classification results saved to {output_file}.")
    except Exception as e:
        logging.error(f"Error saving results to {output_file}: {e}")

# Add argument parsing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify images using a trained GoogLeNet model.")
    parser.add_argument("--image_dir", required=True, help="Path to the directory containing images.")
    parser.add_argument("--model_path", default="models/leaf_health_model.pth", help="Path to the trained model file.")
    parser.add_argument("--output_file", default="classification_results.txt", help="File to save classification results.")
    args = parser.parse_args()

    classify_images(args.image_dir, args.model_path, args.output_file)
