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

    parser.add_argument(
        "-t",
        "--type",
        help="Type of experiment, i.e. classification (input 'c') or segmentation (input 's').",
        metavar="name",
        required=True,
        type=str,
    )

    return parser.parse_args()


def generate_hyperparameter_experiments_classification():

    model_names = [
        "EfficientNetB3",
        "EfficientNetB4",
    ]
    patch_sizes = [32, 64, 128, 256, 512]
    optimizers = ["Adam"]
    learning_rates = [1e-3]
    epochs = [50]
    batch_sizes = [8]

    # patch_sizes = [256]
    # optimizers = ["Adam"]
    # learning_rates = [1e-3]
    # epochs = [50]
    # batch_sizes = [8]

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


def generate_hyperparameter_experiments_segmentation():

    model_names = ["FCN", "DeepLabV3"]
    patch_sizes = [32, 64, 128, 256, 512]
    optimizers = ["Adam"]
    learning_rates = [1e-3]
    epochs = [50]
    batch_sizes = [8]
    loss_fn = ["dice"]

    experiments = list(
        itertools.product(
            patch_sizes,
            optimizers,
            learning_rates,
            epochs,
            batch_sizes,
            model_names,
            loss_fn,
        )
    )

    all_experiments = []

    for i, e in enumerate(experiments):
        experiment_values = list(e)
        json_value = {
            "experiment_number": i,
            "model_name": e[5],
            "batch_size": experiment_values[4],
            "data": f"/space/ariyanzarei/charcoal_dry_rot/datasets/h5_datasets/2022-04-18_{e[0]}X{e[0]}/segmentation_dataset.h5",
            "optimizer": experiment_values[1],
            "learning_rate": experiment_values[2],
            "pre_trained": True,
            "epochs": experiment_values[3],
            "loss_function": experiment_values[6],
        }
        all_experiments.append(json_value)

    print(":: Number of total experiments: " + str(len(experiments)))
    return all_experiments


def main():
    args = get_args()

    if args.type == "c":
        experiments = generate_hyperparameter_experiments_classification()
    elif args.type == "s":
        experiments = generate_hyperparameter_experiments_segmentation()
    else:
        print(":: Invalid type provided. ")
        return

    df = pd.DataFrame.from_records(experiments)
    df["version"] = None
    df["validation_accuracy"] = None
    df["validation_precision"] = None
    df["validation_recall"] = None
    df["validation_f1"] = None

    df.to_csv(
        os.path.join(args.path, f"{args.name}_config_and_results_file.csv"), index=False
    )


main()
# print(df.head())
# print(len(df))
# print(df[df["experiment_number"] == 12])
