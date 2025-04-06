import os
import zipfile
import requests
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import subprocess


class iNaturalistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", batch_size=32, num_workers=7, val_split=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
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

    def get_transformed_dataset(self):
        def calc_transform(dataset, dataset_name):
            channel_sum, channel_squared_sum = torch.zeros(3), torch.zeros(3)
            num_pixels = total_height = total_width = 0

            for img, _ in tqdm(dataset, desc=f"Calculating transform parameters for {dataset_name}", unit="image", leave=False):
                img_tensor = transforms.ToTensor()(img)  # Convert image to tensor
                channel_sum += img_tensor.sum(dim=(1, 2))  # Sum pixel values per channel
                channel_squared_sum += (img_tensor ** 2).sum(dim=(1, 2))  # Sum squared pixel values per channel
                num_pixels += img_tensor.shape[1] * img_tensor.shape[2]  # Total number of pixels per channel

                total_height += img_tensor.shape[1]
                total_width += img_tensor.shape[2]

            mean = channel_sum / num_pixels
            std = torch.sqrt(channel_squared_sum / num_pixels - mean ** 2)

            avg_height = total_height // len(train_dataset)
            avg_width = total_width // len(train_dataset)

            return transforms.Compose([
                transforms.Resize((avg_height, avg_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4713, 0.4598, 0.3893], std=std),
            ])

        train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "train"))
        train_dataset.transform = calc_transform(train_dataset, "train")

        test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "test"))
        test_dataset.transform = calc_transform(test_dataset, "test")

        return train_dataset, test_dataset
    
    def setup(self, stage=None):
        # Load the dataset
        train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "train"), transform=self.transform)
        test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "test"), transform=self.transform)

        # Split the training dataset into training and validation sets
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)),
            test_size=self.val_split,
            stratify=[train_dataset.targets[i] for i in range(len(train_dataset))],
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