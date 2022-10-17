from dataset import DryRotDataset
import torchvision.models as models
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


class UNET:
    def __init__(self, pre_trained=False):
        self.model = models.resnet18(pre_trained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)


class FCN:
    def __init__(self, pre_trained=False):
        self.model = models.resnet18(pre_trained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)


class DeepLabV3:
    def __init__(self, pre_trained=False):
        self.model = models.resnet18(pre_trained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)


class LRASPP:
    def __init__(self, pre_trained=False):
        self.model = models.resnet18(pre_trained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)


# hparams = {
#     "model_name": "ResNet",
#     "batch_size": 8,
#     "data": "/space/ariyanzarei/charcoal_dry_rot/datasets/h5_datasets/2022-04-18_128X128/classification_dataset.h5",
#     "optimizer": "Adam",
#     "learning_rate": 0.0005,
#     "pre_trained": True,
#     "gpu": 1,
#     "epochs": 10,
# }
# model = EfficientNetB4(hparams)
# print(model.model)
