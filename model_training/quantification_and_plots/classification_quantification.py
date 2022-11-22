import cv2
import os
import numpy as np
import sys
import json
import pandas as pd
import argparse
import sys

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


def mask_contains_dryrot(mask):
    return np.max(mask) == 1


def quantify_single_image(image_path, label_path, model, save_path, p_size):
    patches, img = generate_patches(image_path, label_path, p_size, p_size)

    GT_mask = img.copy()
    PR_mask = img.copy()

    total = len(patches.keys())
    gt_dryrot_count = 0.0
    pr_dryrot_count = 0.0

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in patches:

        if (
            patches[i]["mask"].shape[0]
            != patches[i]["row_end"] - patches[i]["row_start"]
            or patches[i]["mask"].shape[1]
            != patches[i]["col_end"] - patches[i]["col_start"]
        ):
            continue

        GT_label = False
        PR_label = False

        if mask_contains_dryrot(patches[i]["mask"]):
            GT_label = True
            gt_dryrot_count += 1
            GT_mask = cv2.rectangle(
                GT_mask,
                (patches[i]["col_start"], patches[i]["row_start"]),
                (patches[i]["col_end"], patches[i]["row_end"]),
                (0, 255, 0),
                -1,
            )

        classification_prediction = model.predict_single_image(patches[i]["patch"])
        if classification_prediction == 1:
            PR_label = True
            pr_dryrot_count += 1
            PR_mask = cv2.rectangle(
                PR_mask,
                (patches[i]["col_start"], patches[i]["row_start"]),
                (patches[i]["col_end"], patches[i]["row_end"]),
                (255, 0, 0),
                -1,
            )

        TP += 1 if GT_label and PR_label else 0
        FP += 1 if not GT_label and PR_label else 0
        TN += 1 if not GT_label and not PR_label else 0
        FN += 1 if GT_label and not PR_label else 0

    alpha = 0.4

    GT_mask = cv2.addWeighted(GT_mask, alpha, img, 1 - alpha, 0)
    PR_mask = cv2.addWeighted(PR_mask, alpha, img, 1 - alpha, 0)

    GT_mask = cv2.cvtColor(GT_mask, cv2.COLOR_RGB2BGR)
    PR_mask = cv2.cvtColor(PR_mask, cv2.COLOR_RGB2BGR)

    cv2.imwrite(
        os.path.join(save_path, image_path.split("/")[-1].replace(".JPG", "_GT.JPG")),
        GT_mask,
    )
    cv2.imwrite(
        os.path.join(save_path, image_path.split("/")[-1].replace(".JPG", "_PR.JPG")),
        PR_mask,
    )

    gt_quantification = gt_dryrot_count / total
    pr_quantification = pr_dryrot_count / total

    print(
        f":: Img: {image_path.split('/')[-1]} GT: {gt_quantification}, PR: {pr_quantification}"
    )
    sys.stdout.flush()

    return {
        "Img": image_path.split("/")[-1],
        "GT": gt_quantification,
        "PR": pr_quantification,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
    }


def save_results(path, ds):
    if os.path.exists(path):
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
    print(f":: Total images: {len(files)}")

    results = []
    for i, file in enumerate(files):
        # if "IMG_0228" not in file:
        #     continue
        print("----------------------------------")
        print(f":: {i}/{len(files)}")
        result = quantify_single_image(
            os.path.join(path_images, f"{file}.JPG"),
            os.path.join(path_image_labels, f"{file}.png"),
            model,
            save_path,
            p_size,
        )
        results.append(result)

    result_array = np.array(
        [[r["GT"], r["PR"], r["TP"], r["FP"], r["TN"], r["FN"]] for r in results]
    )

    with open(os.path.join(save_path, "log.json"), "w+") as f:
        json.dump(result, f)

    TP = np.sum(result_array[:, 2])
    FP = np.sum(result_array[:, 3])
    TN = np.sum(result_array[:, 4])
    FN = np.sum(result_array[:, 5])

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
            "Model-": model_name,
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

    save_path = os.path.join(args.output_path, f"{hparams['model_name']}-{p_size}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    results_path = os.path.join(args.output_path, "results.csv")

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
