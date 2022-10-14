import os
import sys
import argparse
import json

sys.path.append("../..")
sys.path.append("..")
from model_training.models import *
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


def get_args():

    parser = argparse.ArgumentParser(
        description="Training the models using a set of hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--hyperparameters",
        help="The path to the json file that contains the hyperparameters.",
        metavar="hyperparameters",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="The path to the directory where the model checkpoints will be saved. A directory will be created for model class if not exists.",
        metavar="output",
        required=False,
        default="/space/ariyanzarei/charcoal_dry_rot/models/model_checkpoints",
    )

    return parser.parse_args()


def read_hparams(path):
    with open(path, "r") as f:
        hparams = json.load(f)
    return hparams


def train(debug):
    args = get_args()
    hparams = read_hparams(args.hyperparameters)

    chkpt_path = os.path.join(args.output, hparams["model_name"])
    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    model_class = eval(hparams["model_name"])
    model_object = model_class(hparams)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    if debug:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=hparams["gpu"],
            max_epochs=hparams["epochs"],
            callbacks=[checkpoint_callback],
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=chkpt_path,
            accelerator="gpu",
            devices=hparams["gpu"],
            max_epochs=hparams["epochs"],
            callbacks=[checkpoint_callback],
        )

    trainer.fit(
        model_object, model_object.train_dataloader(), model_object.val_dataloader()
    )


train(False)
