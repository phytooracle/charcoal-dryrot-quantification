import os
import random
import numpy as np
import sys
import multiprocessing
import math
import argparse
import cv2
import h5py
from skimage.io import imread
from skimage.transform import resize

PROB_NON_DRYROT = 0.2


def get_args():

    parser = argparse.ArgumentParser(
        description="Generating dataset from the patches of the dry rot images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="The path to the directory in which the dataset files will be created.",
        metavar="output",
        required=True,
    )

    parser.add_argument(
        "-i",
        "--input",
        help="The path to the directory that contains training, validation and test set directories (of patches).",
        metavar="input",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--size",
        help="The final patch size in the h5 file.",
        metavar="size",
        type=int,
        required=True,
    )

    return parser.parse_args()


def get_data_single(args):
    img_path = args[0]
    mask_path = args[1]
    img_name = args[2]
    final_patch_size = args[3]
    channels = args[4]
    size_factor = args[5]

    img = imread(img_path + "/" + img_name)[:, :, :channels]
    img = resize(
        img,
        (final_patch_size * size_factor, final_patch_size * size_factor),
        mode="constant",
        preserve_range=True,
    )

    mask = imread(mask_path + "/" + img_name)[:, :, :channels]
    mask = resize(
        mask,
        (final_patch_size * size_factor, final_patch_size * size_factor),
        mode="constant",
        preserve_range=True,
    )

    images = []
    masks = []
    for i in range(size_factor):
        for j in range(size_factor):
            sub_image = img[
                i * final_patch_size : (i + 1) * final_patch_size,
                j * final_patch_size : (j + 1) * final_patch_size,
                :,
            ]
            sub_mask = mask[
                i * final_patch_size : (i + 1) * final_patch_size,
                j * final_patch_size : (j + 1) * final_patch_size,
                :,
            ]
            if np.max(sub_mask) == 0 and random.random() >= PROB_NON_DRYROT:
                continue
            sub_mask = np.expand_dims(np.max(sub_mask, axis=-1), axis=-1)
            images.append(sub_image)
            masks.append(sub_mask)

    return images, masks


def get_single_image_size(img_name):
    img = cv2.imread(img_name)
    return img.shape[0]


def get_data_and_form_training_val_lists(path, img_size, IMG_CHANNELS):

    training_img_path = os.path.join(path, "training/images")
    training_mask_path = os.path.join(path, "training/annotation")
    validation_img_path = os.path.join(path, "validation/images")
    validation_mask_path = os.path.join(path, "validation/annotation")
    test_img_path = os.path.join(path, "test/images")
    test_mask_path = os.path.join(path, "test/annotation")

    training_images = os.listdir(training_img_path)
    validation_images = os.listdir(validation_img_path)
    test_images = os.listdir(test_img_path)

    actual_size = get_single_image_size(
        os.path.join(training_img_path, training_images[0])
    )
    n_subpatches = int(actual_size / img_size) ** 2

    train_n = len(training_images) * n_subpatches
    validation_n = len(validation_images) * n_subpatches
    test_n = len(test_images) * n_subpatches

    X_train = np.zeros((train_n, img_size, img_size, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((train_n, img_size, img_size, 1), dtype=np.bool)

    X_val = np.zeros((validation_n, img_size, img_size, IMG_CHANNELS), dtype=np.uint8)
    Y_val = np.zeros((validation_n, img_size, img_size, 1), dtype=np.bool)

    X_test = np.zeros((test_n, img_size, img_size, IMG_CHANNELS), dtype=np.uint8)
    Y_test = np.zeros((test_n, img_size, img_size, 1), dtype=np.bool)

    # ----------------- training --------------------

    args = []

    for f in training_images:
        args.append(
            (
                training_img_path,
                training_mask_path,
                f,
                img_size,
                IMG_CHANNELS,
                int(actual_size / img_size),
            )
        )

    print(":: {0} training images to process.".format(len(args)))

    processes = multiprocessing.Pool(int(math.floor(multiprocessing.cpu_count() * 0.8)))
    results = processes.map(get_data_single, args)
    processes.close()

    print(":: {0} training images processed.".format(len(results)))

    ind = 0

    for images, masks in results:
        for i, img in enumerate(images):
            mask = masks[i]
            X_train[ind] = img
            Y_train[ind] = mask
            ind += 1

    X_train = X_train[:ind]
    Y_train = Y_train[:ind]

    ones_count = np.count_nonzero(np.max(Y_train, (1, 2)))
    print(
        ":: label counts: 0 = {0} - 1 = {1}".format(
            Y_train.shape[0] - ones_count, ones_count
        )
    )

    # ----------------- validation --------------------

    args = []

    for f in validation_images:
        args.append(
            (
                validation_img_path,
                validation_mask_path,
                f,
                img_size,
                IMG_CHANNELS,
                int(actual_size / img_size),
            )
        )

    print(":: {0} validation images to process.".format(len(args)))

    processes = multiprocessing.Pool(int(math.floor(multiprocessing.cpu_count() * 0.8)))
    results = processes.map(get_data_single, args)
    processes.close()

    print(":: {0} validation images processed.".format(len(results)))

    ind = 0

    for images, masks in results:
        for i, img in enumerate(images):
            mask = masks[i]
            X_val[ind] = img
            Y_val[ind] = mask
            ind += 1

    X_val = X_val[:ind]
    Y_val = Y_val[:ind]

    ones_count = np.count_nonzero(np.max(Y_val, (1, 2)))
    print(
        ":: label counts: 0 = {0} - 1 = {1}".format(
            Y_val.shape[0] - ones_count, ones_count
        )
    )
    # ----------------- test --------------------

    args = []

    for f in test_images:
        args.append(
            (
                test_img_path,
                test_mask_path,
                f,
                img_size,
                IMG_CHANNELS,
                int(actual_size / img_size),
            )
        )

    print(":: {0} test images to process.".format(len(args)))

    processes = multiprocessing.Pool(int(math.floor(multiprocessing.cpu_count() * 0.8)))
    results = processes.map(get_data_single, args)
    processes.close()

    print(":: {0} test images processed.".format(len(results)))

    ind = 0

    for images, masks in results:
        for i, img in enumerate(images):
            mask = masks[i]
            X_test[ind] = img
            Y_test[ind] = mask
            ind += 1

    X_test = X_test[:ind]
    Y_test = Y_test[:ind]

    ones_count = np.count_nonzero(np.max(Y_test, (1, 2)))
    print(
        ":: label counts: 0 = {0} - 1 = {1}".format(
            Y_test.shape[0] - ones_count, ones_count
        )
    )

    print(":: Database generated successfully...")
    print("------------------------------------------------------")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def generate_segmentation_dataset(path, X_train, Y_train, X_val, Y_val, X_test, Y_test):

    with h5py.File(path, "w") as f:
        f.create_dataset("X_train", data=X_train, dtype="uint8")
        f.create_dataset("Y_train", data=Y_train, dtype="uint8")
        f.create_dataset("X_val", data=X_val, dtype="uint8")
        f.create_dataset("Y_val", data=Y_val, dtype="uint8")
        f.create_dataset("X_test", data=X_test, dtype="uint8")
        f.create_dataset("Y_test", data=Y_test, dtype="uint8")


def generate_classification_dataset(
    path, X_train, Y_train, X_val, Y_val, X_test, Y_test
):

    Y_train = np.reshape(
        Y_train,
        (Y_train.shape[0], Y_train.shape[1] * Y_train.shape[2] * Y_train.shape[3]),
    )
    Y_train = np.max(Y_train, axis=1)

    Y_val = np.reshape(
        Y_val, (Y_val.shape[0], Y_val.shape[1] * Y_val.shape[2] * Y_val.shape[3])
    )
    Y_val = np.max(Y_val, axis=1)

    Y_test = np.reshape(
        Y_test, (Y_test.shape[0], Y_test.shape[1] * Y_test.shape[2] * Y_test.shape[3])
    )
    Y_test = np.max(Y_test, axis=1)

    with h5py.File(path, "w") as f:
        f.create_dataset("X_train", data=X_train, dtype="uint8")
        f.create_dataset("Y_train", data=Y_train, dtype="uint8")
        f.create_dataset("X_val", data=X_val, dtype="uint8")
        f.create_dataset("Y_val", data=Y_val, dtype="uint8")
        f.create_dataset("X_test", data=X_test, dtype="uint8")
        f.create_dataset("Y_test", data=Y_test, dtype="uint8")


def load_dataset(path, ds_type, count=None):
    if ds_type != "segmentation" and ds_type != "classification":
        print(":: Invalid dataset type. Enter either segmentation or classification.")
        return

    with h5py.File(os.path.join(path, f"{ds_type}_dataset.h5"), "r") as f:
        dataset = {}
        for k in f.keys():
            if count is None:
                dataset[k] = f[k][:]
            else:
                dataset[k] = f[k][:count]

    return dataset


def main_data_gen():

    args = get_args()
    if args.input[-1] == "/":
        args.input = args.input[:-1]

    label_date = "{0}_{1}X{1}".format(
        args.input.split("/")[-1].split("_")[0], args.size
    )

    output_path = os.path.join(args.output, "h5_datasets", label_date)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
    ) = get_data_and_form_training_val_lists(args.input, args.size, 3)

    generate_segmentation_dataset(
        os.path.join(output_path, "segmentation_dataset.h5"),
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
    )
    generate_classification_dataset(
        os.path.join(output_path, "classification_dataset.h5"),
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
    )


if __name__ == "__main__":
    main_data_gen()
