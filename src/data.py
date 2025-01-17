import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_dir, batch_size=32):
    logging.info("Starting the data loading process...")

    # Define transformations
    logging.info("Defining data transformations...")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load the dataset
    logging.info(f"Loading dataset from directory: {data_dir}")
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    logging.info(f"Total number of samples in the dataset: {len(dataset)}")
    logging.info(f"Classes found: {dataset.classes}")

    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    logging.info(f"Splitting dataset: {train_size} training samples, {val_size} validation samples, {test_size} test samples.")
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders
    logging.info(f"Creating DataLoaders with batch size: {batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info("Data loading process completed successfully.")
    return train_loader, val_loader, test_loader

# Example usage (remove or comment this before integrating with your main project)
if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data(data_dir="data", batch_size=32)
    logging.info("Train loader, validation loader, and test loader are ready.")
