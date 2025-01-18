import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, transform=None):
    """
    Load the dataset and split it into training, validation, and test sets.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.
        train_ratio (float): Proportion of the dataset to include in the training set.
        val_ratio (float): Proportion of the dataset to include in the validation set.
        transform (callable, optional): Transformations to apply to the images.

    Returns:
        DataLoader: DataLoaders for training, validation, and testing.
    """
    logging.info("Starting the data loading process...")

    # Use default transformations if none are provided
    if transform is None:
        logging.info("Defining default data transformations...")
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    # Load the dataset
    logging.info(f"Loading dataset from directory: {data_dir}")
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None, None, None
    
    logging.info(f"Total number of samples in the dataset: {len(dataset)}")
    logging.info(f"Classes found: {dataset.classes}")

    # Split dataset into train, validation, and test sets
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
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
