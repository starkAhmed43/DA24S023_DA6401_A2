import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, Precision, Recall


class CNNModel(pl.LightningModule):
    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        img_height=32,  # Pass image height as a parameter
        img_width=32,   # Pass image width as a parameter
        conv_filters=[32, 64, 128, 256, 512],
        filter_sizes=[3, 3, 3, 3, 3],
        activation_fn=nn.ReLU,  # Pass the activation function class
        dense_neurons=256,
        batch_norm=False,
        dropout=0.2,
        learning_rate=1e-3,
    ):
        super(CNNModel, self).__init__()
        self.conv_filters = conv_filters
        self.filter_sizes = filter_sizes
        self.activation_fn = activation_fn  # Store the activation function class
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

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.activation_fn()(self.fc1(x))  # Instantiate the activation function
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Log metrics
        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy(y_hat, y))
        self.log("train_precision", self.train_precision(y_hat, y))
        self.log("train_recall", self.train_recall(y_hat, y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Log metrics
        self.log("val_loss", loss)
        self.log("val_accuracy", self.val_accuracy(y_hat, y))
        self.log("val_precision", self.val_precision(y_hat, y))
        self.log("val_recall", self.val_recall(y_hat, y))

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)