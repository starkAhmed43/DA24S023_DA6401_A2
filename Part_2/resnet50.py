import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import MeanMetric
from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy, Precision, Recall, AUROC, F1Score

class ResNet50FineTune(LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super(ResNet50FineTune, self).__init__()
        self.save_hyperparameters()

        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)

        # Freeze all layers except the last fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the last layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.learning_rate = learning_rate

        # Loss function
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
        return self.model(x)
    
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
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }