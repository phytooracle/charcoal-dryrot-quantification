import cv2
import os
import numpy as np
import sys
import json
import pandas as pd
import argparse
import sys
import socket
import datetime

sys.path.append("..")
from PIL import Image
from models import *


def get_args():
    parser = argparse.ArgumentParser(
        description="Quantifications of the classification models on test set images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--patches_path",
        help="The path to the test set patches. The patch names are used to extract the test set images from images_path folder.",
        metavar="patches_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-i",
        "--images_path",
        help="The path to the directory that contains all the raw images.",
        metavar="images_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-l",
        "--labels_path",
        help="The path to the directory that contains all the raw labels/masks.",
        metavar="labels_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output_path",
        help="The path to the directory where all the result images as well as other result files are saved. ",
        metavar="output_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-e",
        "--experiment_path",
        help="The path to the experiment csv file. ",
        metavar="experiment_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-n",
        "--experiment_number",
        help="The index of the experiment from which the model will be loaded and used in the quantificaiton.",
        metavar="experiment_number",
        required=True,
        type=int,
    )

    parser.add_argument(
        "-c",
        "--checkpoint_path",
        help="The path to where all the model checkpoints are saved under.",
        metavar="checkpoint_path",
        required=False,
        default="/space/ariyanzarei/charcoal_dry_rot/models/model_checkpoints",
        type=str,
    )

    return parser.parse_args()


def generate_patches(img_address, mask_address, width, height):
    annotated_patches = {}

    img = np.asarray(Image.open(img_address))
    mask = cv2.imread(mask_address)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask is None:
        return

    x = 0
    y = 0
    i = 0

    mask = mask / 255
    mask = mask.astype("uint8")

    while x + width < img.shape[1]:
        y = 0

        while y + height < img.shape[0]:
            sub_img = img[y : y + height, x : x + width]
            sub_mask = mask[y : y + height, x : x + width]

            annotated_patches[i] = {
                "patch": sub_img,
                "mask": sub_mask,
                "row_start": y,
                "row_end": y + height,
                "col_start": x,
                "col_end": x + width,
            }

            i += 1
            y += height

        x += width

    return annotated_patches, img


def quantify_single_image(image_path, label_path, model, save_path, p_size):
    list_patch_times = []
    patches, img = generate_patches(image_path, label_path, p_size, p_size)

    total = len(patches.keys()) * p_size * p_size
    gt_dryrot_count = 0.0
    pr_dryrot_count = 0.0

    GT_mask = np.zeros(img.shape)
    PR_mask = np.zeros(img.shape)
    for i in patches:
        gt_mask = patches[i]["mask"]
        start_time = datetime.datetime.now()
        pred = model.predict_single_image(patches[i]["patch"])
        end_time = datetime.datetime.now()
        list_patch_times.append((end_time - start_time).total_seconds())
        segmentation_mask = pred.squeeze().astype("uint8")

        z = np.zeros(gt_mask.shape)

        if (
            gt_mask.shape[0] != patches[i]["row_end"] - patches[i]["row_start"]
            or gt_mask.shape[1] != patches[i]["col_end"] - patches[i]["col_start"]
        ):
            continue

        GT_mask[
            patches[i]["row_start"] : patches[i]["row_end"],
            patches[i]["col_start"] : patches[i]["col_end"],
            :,
        ] = np.stack([z, gt_mask, z], axis=2)
        PR_mask[
            patches[i]["row_start"] : patches[i]["row_end"],
            patches[i]["col_start"] : patches[i]["col_end"],
            :,
        ] = np.stack([segmentation_mask, z, z], axis=2)

        gt_dryrot_count += np.count_nonzero(gt_mask)
        pr_dryrot_count += np.count_nonzero(segmentation_mask)

    # alpha = 0.4

    Intersection = np.sum(np.logical_and(GT_mask[:, :, 1], PR_mask[:, :, 0]))
    Union = np.sum(np.logical_or(GT_mask[:, :, 1], PR_mask[:, :, 0]))
    IoU = Intersection / Union

    TP = np.count_nonzero(np.where((GT_mask[:, :, 1] == 1) & (PR_mask[:, :, 0] == 1)))
    FP = np.count_nonzero(np.where((GT_mask[:, :, 1] == 0) & (PR_mask[:, :, 0] == 1)))
    TN = np.count_nonzero(np.where((GT_mask[:, :, 1] == 0) & (PR_mask[:, :, 0] == 0)))
    FN = np.count_nonzero(np.where((GT_mask[:, :, 1] == 1) & (PR_mask[:, :, 0] == 0)))

    # GT_mask = GT_mask * 255
    # GT_mask = GT_mask.astype("uint8")
    # PR_mask = PR_mask * 255
    # PR_mask = PR_mask.astype("uint8")

    # GT_mask = cv2.addWeighted(GT_mask, alpha, img, 1 - alpha, 0)
    # PR_mask = cv2.addWeighted(PR_mask, alpha, img, 1 - alpha, 0)

    # GT_mask = cv2.cvtColor(GT_mask, cv2.COLOR_RGB2BGR)
    # PR_mask = cv2.cvtColor(PR_mask, cv2.COLOR_RGB2BGR)

    # cv2.imwrite(
    #     os.path.join(save_path, image_path.split("/")[-1].replace(".JPG", "_GT.JPG")),
    #     GT_mask,
    # )
    # cv2.imwrite(
    #     os.path.join(save_path, image_path.split("/")[-1].replace(".JPG", "_PR.JPG")),
    #     PR_mask,
    # )

    gt_quantification = gt_dryrot_count / total
    pr_quantification = pr_dryrot_count / total

    print(
        f":: Img: {image_path.split('/')[-1]}\n\tGT: {gt_quantification}\n\tPR: {pr_quantification}\n\tIoU: {IoU}"
    )
    sys.stdout.flush()

    return {
        "Img": image_path.split("/")[-1],
        "GT": gt_quantification,
        "PR": pr_quantification,
        "IOU": IoU,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "PTimes": list_patch_times,
    }


def save_results(path, ds):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
        df = pd.concat([df, ds])
    else:
        df = pd.DataFrame(ds)
    df.to_csv(path, index=False)


def quantify_all_images(
    path_patches,
    path_images,
    path_image_labels,
    save_path,
    results_path,
    model_name,
    model,
    p_size,
):
    files = os.listdir(path_patches)
    files = list(set([f.split("-")[0] for f in files]))

    results = []
    for i, file in enumerate(files):
        # if "IMG_0228" not in file:
        #     continue
        # if i >= 2:
        #     break
        print("----------------------------------")
        print(f":: {i}/{len(files)}")

        result = quantify_single_image(
            os.path.join(path_images, f"{file}.JPG"),
            os.path.join(path_image_labels, f"{file}.png"),
            model,
            save_path,
            p_size,
        )

        results.append((result))

    result_array = np.array(
        [
            [
                r["GT"],
                r["PR"],
                r["IOU"],
                r["TP"],
                r["FP"],
                r["TN"],
                r["FN"],
                sum(r["PTimes"]),
                sum(r["PTimes"]) / len(r["PTimes"]),
            ]
            for r in results
        ]
    )

    # with open(os.path.join(save_path, "log.json"), "w+") as f:
    #     json.dump(results, f)

    TP = np.sum(result_array[:, 3])
    FP = np.sum(result_array[:, 4])
    TN = np.sum(result_array[:, 5])
    FN = np.sum(result_array[:, 6])

    IOU = TP / (TP + FN)
    Acc = (TP + TN) / (TP + FP + TN + FN)
    Precision = 1 if (TP + FP) == 0 else TP / (TP + FP)
    Recall = 1 if (TP + FN) == 0 else TP / (TP + FN)
    F = (
        1
        if (Precision + Recall) == 0
        else 2 * (Precision * Recall) / (Precision + Recall)
    )

    ds = pd.DataFrame(
        {
            "Model": model_name,
            "PathSize": p_size,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "IOU": IOU,
            "ACC": Acc,
            "Precision": Precision,
            "Recall": Recall,
            "F1": F,
            "PerImageTime": str(
                datetime.timedelta(seconds=np.mean(result_array[:, 7]))
            ),
            "PerPatchTime": str(
                datetime.timedelta(seconds=np.mean(result_array[:, 8]))
            ),
        },
        index=[0],
    )

    save_results(results_path, ds)

    print(
        f"Correlation between the Ground Truth and prediction quantification values: {np.corrcoef(result_array[:,0],result_array[:,1])[0,1]}"
    )
    print(f"IoU between the Ground Truth and prediction masks: {IOU}")
    print(f"Accuracy (Pixlewise): {Acc}")
    print(f"Precision (Pixlewise): {Precision}")
    print(f"Recall (Pixlewise): {Recall}")
    print(f"F-1 (Pixlewise): {F}")


def read_hparams(path, experiment_no):
    df = pd.read_csv(path)
    hparams = df[df["experiment_number"] == experiment_no].to_dict(orient="records")[0]
    return hparams


def load_model(base_path, model_name, version_no):
    checkpoints_path = os.path.join(
        base_path, model_name, "lightning_logs", version_no, "checkpoints"
    )
    checkpoint_name = os.listdir(checkpoints_path)[0]
    print(f"checkpoint name: {checkpoint_name}")
    model = eval(model_name).load_from_checkpoint(
        os.path.join(checkpoints_path, checkpoint_name)
    )
    return model


def main():
    if "laplace" in socket.gethostname():
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args = get_args()
    hparams = read_hparams(args.experiment_path, args.experiment_number)

    print("----------------------------------------------------------------")
    print(
        f":: Running for experiment {args.experiment_number}, {hparams['model_name']} "
    )
    print("----------------------------------------------------------------")

    model_version = f"version_{int(hparams['version'])}"
    model = load_model(args.checkpoint_path, hparams["model_name"], model_version)

    p_size = int(hparams["data"].split("/")[-2].split("_")[1].split("X")[0])

    save_path = os.path.join(
        args.output_path, "{0}-{1}".format(hparams["model_name"], p_size)
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    results_path = os.path.join(
        args.output_path,
        "quantification_logs",
        f"{hparams['model_name']}_size_{p_size}.csv",
    )

    quantify_all_images(
        args.patches_path,
        args.images_path,
        args.labels_path,
        save_path,
        results_path,
        hparams["model_name"],
        model,
        p_size,
    )


main()
