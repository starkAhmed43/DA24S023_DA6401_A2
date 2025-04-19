import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

from resnet50 import ResNet50FineTune
from datamodule import iNaturalistDataModule

seed_everything(42, workers=True)

def test_model(ckpt_path, data_augmentation, lr):

    # Initialize the model
    model = ResNet50FineTune.load_from_checkpoint(ckpt_path)

    # Initialize the data module
    data_module = iNaturalistDataModule(
        data_augmentation=data_augmentation,
        batch_size=256,
        num_workers=20,
    )

    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project="DA6401_A2_Finetune",
        name=f"test_run_LR:{lr}_DATAUG:{data_augmentation}",
    )

    # Initialize the trainer
    trainer = Trainer(
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
    )

    # Test the model
    print("Testing the model...")
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    ckpt_path = "./checkpoints/ResNet50_FineTune_LR:0.001_DATAUG:False.ckpt"
    data_augmentation = False
    lr = 0.001

    # Test the model
    test_model(ckpt_path, data_augmentation, lr)