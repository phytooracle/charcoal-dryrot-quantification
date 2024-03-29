from dataset import DryRotDataset
from utils import DiceLoss, UNET_model
import torchvision.models as models
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    fcn_resnet101,
)
from torchvision.models import ResNet101_Weights
import torch.nn as nn
import pytorch_lightning as pl
import torch
import torchmetrics
from torchmetrics.functional import precision_recall
from torch.utils.data import DataLoader


class BaseClassificationClass(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseClassificationClass, self).__init__()
        self.hparams.update(hparams)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy_fn = torchmetrics.Accuracy()
        self.f1_fn = torchmetrics.F1Score()
        self.save_hyperparameters()

    def _build_dataloader(self, ds_path, set_type, shuff=True):
        dataset = DryRotDataset(ds_path, set_type)
        loader = DataLoader(
            dataset, batch_size=self.hparams["batch_size"], num_workers=2, shuffle=shuff
        )
        return loader

    def train_dataloader(self):
        return self._build_dataloader(self.hparams["data"], "train", shuff=True)

    def val_dataloader(self):
        return self._build_dataloader(self.hparams["data"], "val", shuff=False)

    def configure_optimizers(self):
        if self.hparams["optimizer"].lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams["learning_rate"]
            )
        elif self.hparams["optimizer"].lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams["learning_rate"]
            )
        return optimizer

    def forward(self, x):
        x = x.transpose(1, 3).transpose(2, 3)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(-1)
        prediction = self(x)
        loss = self.loss_fn(prediction, y)
        acc = self.accuracy_fn(prediction, y.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(-1)
        prediction = self(x)
        loss = self.loss_fn(prediction, y)
        acc = self.accuracy_fn(prediction, y.int())
        precision, recall = precision_recall(prediction, y.int())
        f1_score = self.f1_fn(prediction, y.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_prec", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_f1", f1_score, prog_bar=True)
        return loss

    def predict_single_image(self, image):
        data = torch.tensor(image).float()
        self.model.eval()
        with torch.no_grad():
            data = data.unsqueeze(0)
            data = data.transpose(1, 3).transpose(2, 3)
            prediction = torch.sigmoid(self.model(data))
            prediction = (prediction > 0.5).float()
            prediction = int(prediction.cpu().item())
        return prediction


class BaseSegmentationClass(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseSegmentationClass, self).__init__()
        self.hparams.update(hparams)
        if hparams["loss_function"] == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif hparams["loss_function"] == "dice":
            self.loss_fn = DiceLoss()
        self.accuracy_fn = torchmetrics.Accuracy()
        self.f1_fn = torchmetrics.F1Score()
        self.save_hyperparameters()

    def _build_dataloader(self, ds_path, set_type, shuff=True):
        dataset = DryRotDataset(ds_path, set_type)
        loader = DataLoader(
            dataset, batch_size=self.hparams["batch_size"], num_workers=2, shuffle=shuff
        )
        return loader

    def train_dataloader(self):
        return self._build_dataloader(self.hparams["data"], "train", shuff=True)

    def val_dataloader(self):
        return self._build_dataloader(self.hparams["data"], "val", shuff=False)

    def configure_optimizers(self):
        if self.hparams["optimizer"].lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams["learning_rate"]
            )
        elif self.hparams["optimizer"].lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams["learning_rate"]
            )
        return optimizer

    def forward(self, x):
        x = x.transpose(1, 3).transpose(2, 3)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.transpose(1, 3).transpose(2, 3)
        prediction = self(x)
        loss = self.loss_fn(prediction, y)
        acc = self.accuracy_fn(prediction, y.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.transpose(1, 3).transpose(2, 3)
        prediction = self(x)
        loss = self.loss_fn(prediction, y)
        acc = self.accuracy_fn(prediction, y.int())
        precision, recall = precision_recall(prediction, y.int())
        f1_score = self.f1_fn(prediction, y.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_prec", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_f1", f1_score, prog_bar=True)
        return loss

    def predict_single_image(self, image):
        data = torch.tensor(image).float()
        self.model.eval()
        with torch.no_grad():
            data = data.unsqueeze(0)
            prediction = torch.sigmoid(self(data))
            prediction = (prediction > 0.5).float()
            prediction = prediction.permute(0, 2, 3, 1)
            prediction = prediction.squeeze(0).cpu().numpy()
        return prediction


class customMNSmall(nn.Module):
    def __init__(self, pretrained):
        super(customMNSmall, self).__init__()
        num_ftrs = pretrained.classifier[0].in_features
        self.pretrained = pretrained
        self.pretrained.classifier[0] = nn.Linear(num_ftrs, num_ftrs, bias=False)
        self.pretrained.classifier[3] = nn.Linear(num_ftrs, num_ftrs // 2)
        self.extension = nn.Sequential(nn.Linear(num_ftrs // 2, 64), nn.Linear(64, 1))

    def forward(self, x):
        x = self.pretrained(x)
        x = self.extension(x)
        return x


# ------------------------- CLassification Models -------------------------------


class ResNet(BaseClassificationClass):
    def __init__(self, hparams):
        super(ResNet, self).__init__(hparams)

        self.model = models.resnet18(self.hparams["pre_trained"])
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)


class MobileNetV3Small(BaseClassificationClass):
    def __init__(self, hparams):
        super(MobileNetV3Small, self).__init__(hparams)

        self.model = models.mobilenet_v3_small(self.hparams["pre_trained"])
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, 1)


class MobileNetV3SmallCustom(BaseClassificationClass):
    def __init__(self, hparams):
        super(MobileNetV3SmallCustom, self).__init__(hparams)
        pretrained = models.mobilenet_v3_small(self.hparams["pre_trained"])
        self.model = customMNSmall(pretrained)


class MobileNetV3Large(BaseClassificationClass):
    def __init__(self, hparams):
        super(MobileNetV3Large, self).__init__(hparams)
        self.model = models.mobilenet_v3_large(self.hparams["pre_trained"])
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, 1)


class EfficientNetB3(BaseClassificationClass):
    def __init__(self, hparams):
        super(EfficientNetB3, self).__init__(hparams)
        self.model = models.efficientnet_b3(self.hparams["pre_trained"])
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 1)


class EfficientNetB4(BaseClassificationClass):
    def __init__(self, hparams):
        super(EfficientNetB4, self).__init__(hparams)
        self.model = models.efficientnet_b4(self.hparams["pre_trained"])
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 1)


# ------------------------- Segmentation Models -------------------------------


class UNET(BaseSegmentationClass):
    def __init__(self, hparams):
        super(UNET, self).__init__(hparams)
        self.model = UNET_model()


class FCN(BaseSegmentationClass):
    def __init__(self, hparams):
        super(FCN, self).__init__(hparams)
        weights = ResNet101_Weights.IMAGENET1K_V2
        self.model = fcn_resnet101(weights_backbone=weights, num_classes=1)

    def forward(self, x):
        output = super(FCN, self).forward(x)
        return output["out"]


class DeepLabV3(BaseSegmentationClass):
    def __init__(self, hparams):
        super(DeepLabV3, self).__init__(hparams)
        weights = ResNet101_Weights.IMAGENET1K_V2
        self.model = deeplabv3_resnet101(weights_backbone=weights, num_classes=1)

    def forward(self, x):
        output = super(DeepLabV3, self).forward(x)
        return output["out"]


# hparams = {
#     "model_name": "DeepLabV3",
#     "batch_size": 8,
#     "data": "/space/ariyanzarei/charcoal_dry_rot/datasets/h5_datasets/2022-04-18_128X128/classification_dataset.h5",
#     "optimizer": "Adam",
#     "learning_rate": 0.0005,
#     "pre_trained": True,
#     "gpu": 1,
#     "epochs": 10,
#     "loss_function": "bce",
# }
# model = DeepLabV3(hparams)
# print(model.model)
