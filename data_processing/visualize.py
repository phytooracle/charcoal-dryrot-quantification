import h5py
import os
import matplotlib.pyplot as plt


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


ind = 1

dataset_512 = load_dataset(
    "/space/ariyanzarei/charcoal_dry_rot/datasets/h5_datasets/2022-04-18_512X512",
    "segmentation",
    count=ind + 1,
)

dataset_256 = load_dataset(
    "/space/ariyanzarei/charcoal_dry_rot/datasets/h5_datasets/2022-04-18_256X256",
    "segmentation",
    count=(ind + 1),
)

dataset_128 = load_dataset(
    "/space/ariyanzarei/charcoal_dry_rot/datasets/h5_datasets/2022-04-18_128X128",
    "segmentation",
    count=(ind + 1),
)

dataset_64 = load_dataset(
    "/space/ariyanzarei/charcoal_dry_rot/datasets/h5_datasets/2022-04-18_64X64",
    "segmentation",
    count=(ind + 1),
)

dataset_32 = load_dataset(
    "/space/ariyanzarei/charcoal_dry_rot/datasets/h5_datasets/2022-04-18_32X32",
    "segmentation",
    count=(ind + 1),
)

fig, axs = plt.subplots(5, 2)
axs[0, 0].imshow(dataset_512["X_train"][ind])
axs[0, 1].imshow(dataset_512["Y_train"][ind])

axs[1, 0].imshow(dataset_256["X_train"][ind])
axs[1, 1].imshow(dataset_256["Y_train"][ind])

axs[2, 0].imshow(dataset_128["X_train"][ind])
axs[2, 1].imshow(dataset_128["Y_train"][ind])

axs[3, 0].imshow(dataset_64["X_train"][ind])
axs[3, 1].imshow(dataset_64["Y_train"][ind])

axs[4, 0].imshow(dataset_32["X_train"][ind])
axs[4, 1].imshow(dataset_32["Y_train"][ind])

plt.axis("off")
plt.show()
