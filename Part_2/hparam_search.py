import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from resnet50 import ResNet50FineTune
from datamodule import iNaturalistDataModule

seed_everything(42, workers=True)

sweep_config = {
    "method": "bayes",  # Bayesian optimization
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [1e-3, 1e-4, 1e-5]},
        "data_augmentation": {"values": [True, False]},
    },
}

def finetune_resnet50(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Initialize the data module
        data_module = iNaturalistDataModule(data_augmentation=config.data_augmentation, batch_size=256, num_workers=20)

        # Initialize the model
        model = ResNet50FineTune(learning_rate=config.learning_rate)

        # Initialize WandB logger
        wandb_logger = WandbLogger(project="DA6401_A2_Finetune")

        # Initialize the trainer
        trainer = Trainer(
            max_epochs=20,
            logger=wandb_logger,
            accelerator="auto",
            devices="auto",
            enable_checkpointing=False,
        )

        # Train the model
        trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="DA6401_A2_Finetune")
    wandb.agent(sweep_id, finetune_resnet50, count=100)