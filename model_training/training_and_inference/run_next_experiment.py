import GPUtil
import pandas as pd
import subprocess
import argparse


def get_args():

    parser = argparse.ArgumentParser(
        description="Running next unfinished experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--path",
        help="The path to the csv file that contains the config of all experiments.",
        metavar="path",
        required=True,
    )

    return parser.parse_args()


def get_next_experiment(path):
    df = pd.read_csv(path)
    na_df = df[df["version"].isna()]
    if len(na_df) == 0:
        return -1

    rec = na_df.iloc[0]
    return rec["experiment_number"]


def main():
    args = get_args()
    path = args.path

    try:
        available_gpu = GPUtil.getFirstAvailable(
            order="first",
            maxLoad=0.01,
            maxMemory=0.01,
            attempts=1,
            interval=900,
            verbose=False,
        )
    except:
        print(":: No GPUs available...")
    else:
        next_experiment = get_next_experiment(path)

        if next_experiment == -1:
            print(":: No more experiments to run.")
        else:
            print(f":: Running experiment {next_experiment} on GPU {available_gpu[0]}")
            process_no = subprocess.Popen(
                " ".join(
                    [
                        "nohup",
                        "python",
                        "train.py",
                        "-g",
                        str(available_gpu[0]),
                        "-p",
                        path,
                        "-e",
                        str(next_experiment),
                        ">/dev/null 2>&1",
                    ]
                ),
                close_fds=True,
                shell=True,
            )


main()
