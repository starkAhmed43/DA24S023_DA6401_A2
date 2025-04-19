import os
import torch
import zipfile
import warnings
import subprocess
from PIL import ImageOps
from tqdm.auto import tqdm
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

def pad_to_square(img, fill=0):
    width, height = img.size
    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,     # left
        (max_side - height) // 2,    # top
        (max_side - width + 1) // 2, # right
        (max_side - height + 1) // 2 # bottom
    )
    return ImageOps.expand(img, padding, fill=fill)

class iNaturalistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="../data", batch_size=128, num_workers=1, val_split=0.2, image_dim=224, data_augmentation=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.image_dim = image_dim
        self.data_augmentation = data_augmentation
        self.base_transform = transforms.Compose([
            transforms.Lambda(pad_to_square),
            transforms.Resize((self.image_dim, self.image_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.2]*3),
        ])
        self.augmentation_transform = transforms.Compose([
            transforms.Lambda(pad_to_square),
            transforms.Resize((self.image_dim, self.image_dim)),
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            transforms.RandomRotation(degrees=15),  # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.2]*3),
        ])

    def prepare_data(self):
        # Download and extract the dataset if it doesn't exist
        if not os.path.exists(self.data_dir):
            zip_path = "iNaturalist.zip"
            url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
            
            if os.path.exists(zip_path):
                print("Zip file already exists, skipping download.")
            else:
                print("Downloading dataset...")
                subprocess.run(["curl", "-o", zip_path, "-L", url], check=True)

            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(".")
            
            os.rename("inaturalist_12K", self.data_dir)  # Rename extracted folder to 'data'

            # Rename 'val' folder to 'test' inside the renamed 'data' directory
            os.rename(os.path.join(self.data_dir, "val"), os.path.join(self.data_dir, "test"))
    
    def setup(self, stage=None):
        # Load the dataset
        train_transform = self.augmentation_transform if self.data_augmentation else self.base_transform
        train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "train"), transform=train_transform)
        test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "test"), transform=self.base_transform)

        # Split the training dataset into training and validation sets
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)),
            test_size=self.val_split,
            stratify=[train_dataset.targets[i] for i in range(len(train_dataset))],
            random_state=42,
        )
        self.train_dataset = Subset(train_dataset, train_indices)
        self.val_dataset = Subset(train_dataset, val_indices)
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self,):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)


if __name__ == "__main__":
    data_module = iNaturalistDataModule()
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")