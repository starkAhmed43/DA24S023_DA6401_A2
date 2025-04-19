import wandb
import torch.nn as nn
from model import CNNModel
from datamodule import iNaturalistDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

def test_model(project_name,
               checkpoint_path,
               lr=0.001,
               data_augmentation=True,
               num_filters=32,
               filter_sizes=3,
               filter_organisation="double",
               activation_fn="ReLU",
               batch_norm=True,
               dropout=0.3,
               dense_neurons=64,
               ):

    if not checkpoint_path:
        print("No checkpoint found on WandB. Ensure the specified run exists and has a checkpoint.")
        return

    print(f"Loading checkpoint from WandB: {checkpoint_path}")

    # Map activation function names to PyTorch classes
    activation_fn_map = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
        "Mish": nn.Mish,
    }

    # Determine convolutional filters based on filter organization
    if filter_organisation == "double":
        conv_filters = [min(num_filters * (2 ** i), 256) for i in range(5)]
    elif filter_organisation == "halve":
        conv_filters = [max(num_filters // (2 ** i), 16) for i in range(5)]
    else:
        conv_filters = [num_filters] * 5

    # Initialize the model
    model = CNNModel(
        img_height=224,
        img_width=224,
        conv_filters=conv_filters,
        filter_sizes=[filter_sizes] * 5,
        activation_fn=activation_fn_map[activation_fn],
        dense_neurons=dense_neurons,
        num_classes=10,
        batch_norm=batch_norm,
        dropout=dropout,
        learning_rate=lr,
    )

    # Load the checkpoint
    model = CNNModel.load_from_checkpoint(checkpoint_path)

    # Initialize the data module
    data_module = iNaturalistDataModule(
        image_dim=224,
        val_split=0.2,
        data_augmentation=data_augmentation,
        batch_size=256,
        num_workers=32,
    )

    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project="DA6401_A2",
        name=f"test_run_LR:{lr}_DATAUG:{data_augmentation}_FILTERS:{num_filters}_FILTERSIZE:{filter_sizes}_FILTERORG:{filter_organisation}_ACTIVATION:{activation_fn}_BATCHNORM:{batch_norm}_DROPOUT:{dropout}_DENSE:{dense_neurons}",
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
    # Specify the project name and run name
    project_name = "DA6401_A2"
    ckpt_path = "./checkpoints/train_run_LR_0.0001_DATAUG:True_FILTERS:64_FILTERSIZE:5_FILTERORG:double_ACTIVATION:Mish_BATCHNORM:True_DROPOUT:0.3_DENSE:32.ckpt"
    # Test the model
    test_model(project_name, ckpt_path)