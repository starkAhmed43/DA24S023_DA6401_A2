import wandb
import torch.nn as nn
from model import CNNModel
from datamodule import iNaturalistDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

seed_everything(42, workers=True)

def get_best_config(project_name):
    # Initialize WandB API
    api = wandb.Api()

    # Fetch all runs for the project
    runs = api.runs(project_name)

    # Find the run with the maximum val_accuracy
    best_run = max(runs, key=lambda run: run.summary.get("val_f1", 0))

    # Return the best configuration
    return dict(best_run.config)

def train_with_best_config(best_config):
    # Map activation function names to PyTorch classes
    activation_fn_map = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
        "Mish": nn.Mish,
    }

    # Determine convolutional filters based on filter organization
    if best_config["filter_organisation"] == "double":
        conv_filters = [min(best_config["num_filters"] * (2 ** i), 256) for i in range(5)]
    elif best_config["filter_organisation"] == "halve":
        conv_filters = [max(best_config["num_filters"] // (2 ** i), 16) for i in range(5)]
    else:
        conv_filters = [best_config["num_filters"]] * 5

    # Initialize the model
    model = CNNModel(
        img_height=224,
        img_width=224,
        conv_filters=conv_filters,
        filter_sizes=[best_config["filter_sizes"]] * 5,
        activation_fn=activation_fn_map[best_config["activation_fn"]],
        dense_neurons=best_config["dense_neurons"],
        num_classes=10,
        batch_norm=best_config["batch_norm"],
        dropout=best_config["dropout"],
        learning_rate=best_config["learning_rate"],
    )

    # Initialize the data module
    data_module = iNaturalistDataModule(
        image_dim=224,
        val_split=0.2,
        data_augmentation=best_config["data_augmentation"],
        batch_size=256,
        num_workers=32,
    )

    name=(
        f"train_run_LR_{best_config['learning_rate']}_"
        f"DATAUG:{best_config['data_augmentation']}_"
        f"FILTERS:{best_config['num_filters']}_"
        f"FILTERSIZE:{best_config['filter_sizes']}_"
        f"FILTERORG:{best_config['filter_organisation']}_"
        f"ACTIVATION:{best_config['activation_fn']}_"
        f"BATCHNORM:{best_config['batch_norm']}_"
        f"DROPOUT:{best_config['dropout']}_"
        f"DENSE:{best_config['dense_neurons']}"
    )

    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project="DA6401_A2",
        name=name,
    )

    # Initialize the model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename=name,
        save_top_k=1,
        monitor="val_f1",
        mode="max",
    )

    # Initialize the trainer
    trainer = Trainer(
        max_epochs=30,
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Log the best model to WandB
    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path:  # Ensure a checkpoint was saved
        artifact = wandb.Artifact(name.replace(":", "-"), type="model")
        artifact.add_file(best_checkpoint_path)
        wandb.log_artifact(artifact)
        print(f"Uploaded checkpoint to WandB: {best_checkpoint_path}")
    else:
        print("No checkpoint was saved.")

if __name__ == "__main__":
    # Specify the project name
    project_name = "DA6401_A2"

    # Get the best configuration
    best_config = get_best_config(project_name)

    # Train the model with the best configuration
    train_with_best_config(best_config)