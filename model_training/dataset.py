import torch
import numpy as np
import torch.utils.data as data
import h5py


class DryRotDataset(data.Dataset):
    def __init__(self, h5_filename, set_type):
        super().__init__()
        self.h5_filename = h5_filename
        self.length = -1
        self.set = set_type

    def __getitem__(self, index):
        f = h5py.File(self.h5_filename, "r")
        image = f[f"X_{self.set}"][index]
        label = f[f"Y_{self.set}"][index]

        f.close()

        return (
            torch.tensor(image).float(),
            torch.tensor(label).float(),
        )

    def __len__(self):
        if self.length != -1:
            return self.length
        else:
            f = h5py.File(self.h5_filename, "r")
            self.length = f[f"X_{self.set}"].shape[0]
            f.close()
            return self.length


# ds = DryRotDataset(
#     "/space/ariyanzarei/charcoal_dry_rot/datasets/h5_datasets/2022-04-18_32X32/segmentation_dataset.h5",
#     "train",
# )
# print(ds[0][1].shape)
