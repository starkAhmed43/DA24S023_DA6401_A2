import wandb
import torch.nn as nn
from model import CNNModel
from datamodule import iNaturalistDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

sweep_config = {
    "method": "bayes",  # Bayesian optimization
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "num_filters": {
            "values": [16, 32, 64]
        },
        "filter_sizes": {
            "values": [3, 5]
        },
        "filter_organisation": {
            "values": ["same", "double", "halve"]
        },
        "activation_fn": {"values": ["ReLU", "GELU", "SiLU", "Mish"]},
        "learning_rate": {"values": [1e-3, 1e-4, 1e-5]},
        "data_augmentation": {"values": [True, False]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.2, 0.3]},
        "dense_neurons": {"values": [32, 64, 128]},
    },
}

def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Map activation function names to PyTorch classes
        activation_fn_map = {
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "SiLU": nn.SiLU,
            "Mish": nn.Mish,
        }
        conv_filters = []            
        if config.filter_organisation == "double":
            conv_filters = [min(config.num_filters * (2 ** i), 256) for i in range(5)]
        elif config.filter_organisation == "halve":
            conv_filters = [max(config.num_filters // (2 ** i), 16) for i in range(5)]
        else:
            conv_filters = [config.num_filters] * 5

        # Initialize the model
        model = CNNModel(
            img_height=224,
            img_width=224,
            conv_filters=conv_filters,
            filter_sizes=[config.filter_sizes] * 5,
            activation_fn=activation_fn_map[config.activation_fn],
            dense_neurons=config.dense_neurons,
            num_classes=10,
            batch_norm=config.batch_norm,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
        )

        # Initialize the data module
        data_module = iNaturalistDataModule(
            image_dim=224,
            val_split=0.2,
            data_augmentation=config.data_augmentation,
            batch_size=256, 
            num_workers=32
        )

        # Initialize WandB logger
        wandb_logger = WandbLogger(project="DA6401_A2")

        # Train the model
        trainer = Trainer(
            max_epochs=30,
            logger=wandb_logger,
            accelerator="auto",
            devices="auto",
            enable_checkpointing=False,
        )
        trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="DA6401_A2")
    wandb.agent(sweep_id, train_model, count=30)