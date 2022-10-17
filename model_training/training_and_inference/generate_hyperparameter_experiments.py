import itertools
import json
import socket
import os
import pandas as pd
import argparse


def get_args():

    parser = argparse.ArgumentParser(
        description="Generating hyperparameters experiment sets. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--path",
        help="The path to which the hyperparameter experiment json file will be saved.",
        metavar="path",
        required=True,
    )

    parser.add_argument(
        "-n",
        "--name",
        help="The name of the experiment",
        metavar="name",
        required=True,
    )

    return parser.parse_args()


def generate_hyperparameter_experiments_classification():

    model_names = [
        "EfficientNetB3",
        "MobileNetV3Large",
        "MobileNetV3Small",
        "EfficientNetB4",
        "MobileNetV3SmallCustom",
        "ResNet",
    ]
    # patch_sizes = [32, 64, 128, 256, 512]
    # optimizers = ["Adam"]
    # learning_rates = [1e-2, 1e-3, 1e-4]
    # epochs = [1]
    # batch_sizes = [4, 8, 16]

    patch_sizes = [256]
    optimizers = ["Adam"]
    learning_rates = [1e-3]
    epochs = [50]
    batch_sizes = [8]

    experiments = list(
        itertools.product(
            patch_sizes, optimizers, learning_rates, epochs, batch_sizes, model_names
        )
    )

    all_experiments = []

    for i, e in enumerate(experiments):
        experiment_values = list(e)
        json_value = {
            "experiment_number": i,
            "model_name": e[5],
            "batch_size": experiment_values[4],
            "data": f"/space/ariyanzarei/charcoal_dry_rot/datasets/h5_datasets/2022-04-18_{e[0]}X{e[0]}/classification_dataset.h5",
            "optimizer": experiment_values[1],
            "learning_rate": experiment_values[2],
            "pre_trained": True,
            "epochs": experiment_values[3],
        }
        all_experiments.append(json_value)

    print(":: Number of total experiments: " + str(len(experiments)))
    return all_experiments


def main():
    args = get_args()
    experiments = generate_hyperparameter_experiments_classification()

    df = pd.DataFrame.from_records(experiments)
    df["version"] = None
    df["validation_accuracy"] = None
    df["validation_precision"] = None
    df["validation_recall"] = None
    df["validation_f1"] = None

    df.to_csv(os.path.join(args.path, f"{args.name}_config_and_results_file.csv"))


main()
# print(df.head())
# print(len(df))
# print(df[df["experiment_number"] == 12])
