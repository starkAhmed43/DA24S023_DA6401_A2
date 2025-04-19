import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy, Precision, Recall, AUROC, F1Score

class CNNModel(pl.LightningModule):
    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        img_height=224,
        img_width=224,
        conv_filters=[64, 128, 256, 256, 256],
        filter_sizes=[5, 5, 5, 5, 5],
        activation_fn=nn.Mish,
        dense_neurons=32,
        batch_norm=True,
        dropout=0.3,
        learning_rate=1e-3,
    ):
        super(CNNModel, self).__init__()
        self.conv_filters = conv_filters
        self.filter_sizes = filter_sizes
        self.activation_fn = activation_fn
        self.dense_neurons = dense_neurons
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Define convolutional layers
        layers = []
        in_channels = input_channels
        for i in range(len(conv_filters)):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_filters[i],
                    kernel_size=filter_sizes[i],
                    stride=1,
                    padding=filter_sizes[i] // 2,
                )
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(conv_filters[i]))
            layers.append(activation_fn())  # Instantiate the activation function
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = conv_filters[i]

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the output dimensions after convolution and pooling
        final_height = img_height // (2 ** len(conv_filters))
        final_width = img_width // (2 ** len(conv_filters))
        flattened_features = conv_filters[-1] * final_height * final_width

        # Define fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_features, dense_neurons)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_neurons, num_classes)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")

        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.activation_fn()(self.fc1(x))  # Instantiate the activation function
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x
    
    def on_train_start(self):
        self.val_loss.reset()
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()

        self.train_loss.reset()
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
        self.train_auroc.reset()

    def model_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, logits, y

    def training_step(self, batch, batch_idx):
        loss, preds, logits, targets = self.model_step(batch)

        # Update metrics
        self.train_loss(loss)
        self.train_accuracy(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_f1(preds, targets)
        self.train_auroc(logits, targets)

        # Log metrics
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True)
        self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train_recall", self.train_recall, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, logits, targets = self.model_step(batch)

        # Update metrics
        self.val_loss(loss)
        self.val_accuracy(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)
        self.val_auroc(logits, targets)

        # Log metrics
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True)
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, logits, targets = self.model_step(batch)

        # Update metrics
        self.test_loss(loss)
        self.test_accuracy(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)
        self.test_auroc(logits, targets)

        # Log metrics
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }