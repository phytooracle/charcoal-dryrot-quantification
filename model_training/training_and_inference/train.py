import os
import sys
import argparse
import pandas as pd

sys.path.append("../..")
sys.path.append("..")
from model_training.models import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl


def get_args():

    parser = argparse.ArgumentParser(
        description="Training the models using a set of hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--path",
        help="The path to the csv file that contains the config of all experiments.",
        metavar="path",
        required=True,
    )

    parser.add_argument(
        "-e",
        "--experiment",
        help="The number of experiment to run.",
        metavar="experiment",
        required=True,
        type=int,
    )

    parser.add_argument(
        "-g",
        "--gpu",
        help="The gpu index to be used for training.",
        metavar="gpu",
        required=True,
        type=int,
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


def read_hparams(path, experiment_no):
    df = pd.read_csv(path)
    hparams = df[df["experiment_number"] == experiment_no].to_dict(orient="records")[0]
    return hparams


def test_and_save_results(trainer, experiment_path, experiment_no):
    try:
        metrics = trainer.validate(ckpt_path="best")[0]
        df = pd.read_csv(experiment_path)
        df.loc[(df.experiment_number == experiment_no), "version"] = int(
            trainer.logger.version
        )
        df.loc[
            (df.experiment_number == experiment_no), "validation_accuracy"
        ] = metrics["val_acc"]

        df.loc[
            (df.experiment_number == experiment_no), "validation_precision"
        ] = metrics["val_prec"]

        df.loc[(df.experiment_number == experiment_no), "validation_recall"] = metrics[
            "val_recall"
        ]

        df.loc[(df.experiment_number == experiment_no), "validation_f1"] = metrics[
            "val_f1"
        ]

        df.to_csv(experiment_path)

        print(":: Successfully saved results. ")
    except:
        print(":: Error occured. ")


def log_version_in_file(experiment_path, experiment_no, trainer):
    df = pd.read_csv(experiment_path)
    df.loc[(df.experiment_number == experiment_no), "version"] = int(
        trainer.logger.version
    )
    df.to_csv(experiment_path)


def train(debug):
    args = get_args()
    hparams = read_hparams(args.path, args.experiment)

    chkpt_path = os.path.join(args.output, hparams["model_name"])
    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    model_class = eval(hparams["model_name"])
    model_object = model_class(hparams)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=20)

    if debug:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[args.gpu],
            # max_epochs=hparams["epochs"],
            max_epochs=1,
            callbacks=[checkpoint_callback, early_stopping_callback],
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=chkpt_path,
            accelerator="gpu",
            devices=[args.gpu],
            max_epochs=hparams["epochs"],
            callbacks=[checkpoint_callback, early_stopping_callback],
        )

    log_version_in_file(args.path, args.experiment, trainer)

    trainer.fit(
        model_object, model_object.train_dataloader(), model_object.val_dataloader()
    )
    test_and_save_results(trainer, args.path, args.experiment)


train(False)
