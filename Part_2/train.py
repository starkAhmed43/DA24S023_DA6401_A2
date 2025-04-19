import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from resnet50 import ResNet50FineTune
from datamodule import iNaturalistDataModule

seed_everything(42, workers=True)

def get_best_config(project_name):
    # Initialize WandB API
    api = wandb.Api()

    # Fetch all runs for the project
    runs = api.runs(project_name)

    # Find the run with the maximum val_f1
    best_run = max(runs, key=lambda run: run.summary.get("val_f1", 0))

    # Return the best configuration
    return best_run.config

def train_with_best_config(best_config):
    # Initialize the data module
    data_module = iNaturalistDataModule(
        data_augmentation=best_config["data_augmentation"],
        batch_size=256,
        num_workers=20,
    )

    # Initialize the model
    model = ResNet50FineTune(learning_rate=best_config["learning_rate"])

    # Initialize WandB logger
    wandb_logger = WandbLogger(project="DA6401_A2_Finetune", name=f"train_run_LR:{best_config['learning_rate']}_DATAUG:{best_config['data_augmentation']}")

    # Initialize the model checkpoint callback
    # Save the best model based on validation F1 score
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename=f"ResNet50_FineTune_LR:{best_config['learning_rate']}_DATAUG:{best_config['data_augmentation']}",
        save_top_k=1,
        monitor="val_f1",
        mode="max",
    )
    
    # Initialize the trainer
    trainer = Trainer(
        max_epochs=20,
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        enable_checkpointing=True,
        callbacks=[checkpoint_callback]
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Log the best model to WandB
    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path:  # Ensure a checkpoint was saved
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(best_checkpoint_path)
        wandb.log_artifact(artifact)
        print(f"Uploaded checkpoint to WandB: {best_checkpoint_path}")
    else:
        print("No checkpoint was saved.")

if __name__ == "__main__":
    # Specify the project name
    project_name = "DA6401_A2_Finetune"

    # Get the best configuration
    best_config = get_best_config(project_name)

    # Train the model with the best configuration
    train_with_best_config(best_config)