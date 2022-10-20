import GPUtil
import pandas as pd
import subprocess
import argparse
import socket


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
        system = socket.gethostname().split(".")[0]
        order = "last" if system == "laplace" else "first"
        available_gpu = GPUtil.getFirstAvailable(
            order=order,
            maxLoad=0.5,
            maxMemory=0.5,
            attempts=1,
            interval=900,
            verbose=False,
        )
        if system == "laplace" and available_gpu[0] == 0:
            print(
                ":: No GPUs/GPU computation power available. GPU 0 in laplace does not work..."
            )
            return
    except:
        print(":: No GPUs/GPU computation power available...")
    else:
        next_experiment = get_next_experiment(path)

        if next_experiment == -1:
            print(":: No more experiments to run.")
        else:
            output_name = f"Exp_{next_experiment}_GPU_{available_gpu[0]}_{system}.out"
            process_info = subprocess.Popen(
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
                        f"> {output_name} 2>&1 &",
                    ]
                ),
                close_fds=True,
                shell=True,
            )
            print(
                f":: Running experiment {next_experiment} on GPU {available_gpu[0]}, PID: {process_info.pid}."
            )


main()
