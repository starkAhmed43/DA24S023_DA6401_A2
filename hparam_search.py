import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from datamodule import iNaturalistDataModule
from model import CNNModel
import torch.nn as nn

sweep_config = {
    "method": "bayes",  # Bayesian optimization
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "conv_filters": {
            "values": [[32, 32, 32, 32, 32], [32, 64, 128, 256, 512], [512, 256, 128, 64, 32]]
        },
        "activation_fn": {"values": ["ReLU", "GELU", "SiLU", "Mish"]},
        "data_augmentation": {"values": [True, False]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.2, 0.3]},
        "batch_size": {"values": [32, 64]},
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

        # Initialize the model
        model = CNNModel(
            img_height=32,
            img_width=32,
            conv_filters=config.conv_filters,
            activation_fn=activation_fn_map[config.activation_fn],
            batch_norm=config.batch_norm,
            dropout=config.dropout,
        )

        # Initialize the data module
        data_module = iNaturalistDataModule(
            batch_size=config.batch_size,
            val_split=0.2,
        )

        # Initialize WandB logger
        wandb_logger = WandbLogger(project="DA6402_A2")

        # Train the model
        trainer = Trainer(
            max_epochs=30,
            logger=wandb_logger,
            accelerator="auto",
            devices="auto",
        )
        trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="DA6402_A2")
    wandb.agent(sweep_id, train_model, count=5)