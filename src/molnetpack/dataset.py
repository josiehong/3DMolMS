"""
Date: 2023-10-02 20:24:27
LastEditors: yuhhong
LastEditTime: 2023-10-20 17:01:37
"""

import pickle
import numpy as np

from torch.utils.data import Dataset


class MolMS_Dataset(Dataset):
    def __init__(self, x, data_augmentation=True, precursor_type=False, mode="path"):
        if mode == "path":
            path = x
            with open(path, "rb") as file:
                data = pickle.load(file)
        elif mode == "data":
            data = x
            path = "unknown"
        else:
            raise ValueError("Unsupported mode:", mode)

        if precursor_type:
            data = self.filter_precursor_type(data, precursor_type)

        # generate mask
        for idx in range(len(data)):
            mask = ~np.all(data[idx]["mol"] == 0, axis=1)
            data[idx]["mask"] = mask.astype(bool)

        # data augmentation by flipping the x,y,z-coordinates
        if data_augmentation:
            flipping_data = []
            for d in data:
                flipping_mol_arr = np.copy(d["mol"])
                flipping_mol_arr[:, 0] *= -1
                flipping_data.append(
                    {
                        "title": d["title"] + "_f",
                        "mol": flipping_mol_arr,
                        "spec": d["spec"],
                        "env": d["env"],
                    }
                )

            self.data = data + flipping_data
            print(
                "Load {} data (with data augmentation by flipping coordinates)".format(
                    len(self.data)
                )
            )
        else:
            self.data = data
            print("Load {} data".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["spec"],
            self.data[idx]["env"],
        )

    def filter_precursor_type(self, data, precursor_type):
        filtered_data = []
        for d in data:
            d_precursor_type = ",".join([str(int(i)) for i in d["env"][1:]])
            if d_precursor_type == precursor_type:
                filtered_data.append(d)
        return filtered_data


class Mol_Dataset(Dataset):
    def __init__(self, data, precursor_type=False):
        if precursor_type:
            data = self.filter_precursor_type(data, precursor_type)

        # generate mask
        for idx in range(len(data)):
            mask = ~np.all(data[idx]["mol"] == 0, axis=1)
            data[idx]["mask"] = mask.astype(bool)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["env"],
        )

    def filter_precursor_type(self, data, precursor_type):
        filtered_data = []
        for d in data:
            d_precursor_type = ",".join([str(int(i)) for i in d["env"][1:]])
            if d_precursor_type == precursor_type:
                filtered_data.append(d)
        return filtered_data


class MolRT_Dataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as file:
            self.data = pickle.load(file)
        print("Load {} data from {}".format(len(self.data), path))

        # generate mask
        for idx in range(len(self.data)):
            mask = ~np.all(self.data[idx]["mol"] == 0, axis=1)
            self.data[idx]["mask"] = mask.astype(bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["rt"],
        )


class MolCCS_Dataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as file:
            self.data = pickle.load(file)
        print("Load {} data from {}".format(len(self.data), path))

        # generate mask
        for idx in range(len(self.data)):
            mask = ~np.all(self.data[idx]["mol"] == 0, axis=1)
            self.data[idx]["mask"] = mask.astype(bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["ccs"],
            self.data[idx]["env"],
        )


class MolPRE_Dataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as file:
            data = pickle.load(file)

        self.data = []
        for d in data:
            if "mol" in d.keys():
                self.data.append(d)
        print("Load {} data from {}".format(len(self.data), path))

        # generate mask
        for idx in range(len(self.data)):
            mask = ~np.all(self.data[idx]["mol"] == 0, axis=1)
            self.data[idx]["mask"] = mask.astype(bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["y"],
        )


class MolCSV_Dataset(Dataset):
    def __init__(self, x, mode="path"):
        assert mode in ["path", "data"]
        if mode == "path":
            with open(x, "rb") as file:
                self.data = pickle.load(file)
            print("Load {} data from {}".format(len(self.data), x))
        elif mode == "data":
            self.data = x

        # generate mask
        for idx in range(len(self.data)):
            mask = ~np.all(self.data[idx]["mol"] == 0, axis=1)
            self.data[idx]["mask"] = mask.astype(bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["prop"],
        )


class MolCSV_Test_Dataset(Dataset):
    def __init__(self, x, mode="path"):
        assert mode in ["path", "data"]
        if mode == "path":
            with open(x, "rb") as file:
                self.data = pickle.load(file)
            print("Load {} data from {}".format(len(self.data), x))
        elif mode == "data":
            self.data = x

        # generate mask
        for idx in range(len(self.data)):
            mask = ~np.all(self.data[idx]["mol"] == 0, axis=1)
            self.data[idx]["mask"] = mask.astype(bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["title"], self.data[idx]["mol"], self.data[idx]["mask"]
