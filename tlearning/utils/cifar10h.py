"""
==========================================
CIFAR-10H dataset as a torchvision dataset
==========================================

CIFAR-10H dataset
------------------
data
We have the labels for the training (probabilities).
Each label is associated to its corresponding image the test batch of CIFAR-10.

We use a modified version of the CIFAR-10 class-implementation available at
https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py


CIFAR-10 dataset
------------------

There are 5 training batches and 1 test batch.
Each batch includes the following in a dict format:

    data_batch_i, i=1,...,5 and test_batch
    ├── batch_label: the number of the batch
    ├── labels: id true label for each image
    │   ├── 0 -> plane
    |   ├── 1 -> car
    |   ├── 2 -> bird
    |   ├── 3 -> cat
    |   ├── 4 -> deer
    |   ├── 5 -> dog
    |   ├── 6 -> frog
    |   ├── 7 -> horse
    |   ├── 8 -> ship
    │   └── 9 -> truck
    ├── data: (3072, ) numpy arrays as [1024-red | 1024 green | 1024 blue]
    └── filenames: name of the image
"""

import os.path as path
from typing import Any, Callable, Optional, Union, Tuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import zipfile
import torch
import pandas as pd
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

str_bool = Union[str, bool]
path_file = path.dirname(__file__)
path_data = path.join(path_file, "data")


class CIFAR10H(VisionDataset):
    """`CIFAR-10H <https://github.com/jcpeterson/cifar-10h>`_ Dataset.

        Args:
            root (string): Root directory of dataset where directory
                ``cifar-10h-master`` exists or will be saved to if download
                 is set to True.
            train (str/bool, optional): Default="True", the behavior is:
    $           - train=True: uses the 10k images of the train set,
                - train=False: uses the 50k images of the test set.
            transform (callable, optional): Transformations on the PIL images.
            target_transform (callable, optional): A function/transform that takes
                in the target and transforms it.
            download (str/bool, optional): If True, downloads the datasets and puts
                it in root directory. If datasets are already downloaded, it is not
                downloaded again except if "force" is passed. The datasets included
                are both CIFAR-10 and the probabilities from CIFAR-10H
    """

    root = path_data
    base_folder_cifarh = "cifar-10h-master"
    base_folder_cifar = "cifar-10-batches-py"
    url_cifarh = "https://github.com/jcpeterson/cifar-10h/archive/master.zip"
    url_cifar = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename_cifar = "cifar-10-python.tar.gz"
    filename_cifarh = "cifar-10h-master.zip"

    train_list_cifar = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]

    test_list_cifar = ["test_batch"]  # also the train of cifar-10h
    probas_cifarh = [path.join("data", "cifar10h-probs.npy")]
    classes_labels = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    def __init__(
        self,
        root: str,
        train: str_bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: str_bool = False,
    ) -> None:
        super(CIFAR10H, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.train = train
        if self.train:
            files_list = self.test_list_cifar + self.probas_cifarh
        else:
            files_list = self.train_list_cifar

        if download:
            self.download(download)

        self.data: Any = []
        self.true_targets: Any = []
        self.targets: Any = []

        for idx, filename in enumerate(files_list):
            if filename[-3:] != "npy":
                base = self.base_folder_cifar
                file_path = path.join(self.root, base, filename)
                with open(file_path, "rb") as f:
                    entry = pickle.load(f, encoding="latin1")
                    self.data.append(entry["data"])
                    if "labels" in entry:
                        self.true_targets.extend(entry["labels"])
                    else:
                        self.true_targets.extend(entry["fine_labels"])
            else:
                base = self.base_folder_cifarh
                file_path = path.join(self.root, base, filename)
                preds = np.load(file_path)
                self.targets.extend(preds.tolist())

        self.targets = [
            torch.tensor(tar, dtype=torch.float).view(1, -1)
            for tar in self.targets
        ]
        if not train:
            self.targets = self.true_targets
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(
            (0, 2, 3, 1)
        )  # convert to HWC self.data

    def download(self, download) -> None:
        """Checks and downloads the CIFAR10 and CIFAR10H files.

        Args:
            download (str/bool): Force the download or simply check the files
        """
        path_cifarh = path.join(self.root, self.filename_cifarh)
        path_cifar = path.join(self.root, self.filename_cifar)
        is_there = path.isfile(path_cifarh) and path.isfile(path_cifar)
        if is_there:
            print("Files already exist.")
        if download == "force" or not is_there:
            download_and_extract_archive(
                self.url_cifar, self.root, filename=self.filename_cifar
            )
            download_and_extract_archive(
                self.url_cifarh, self.root, filename=self.filename_cifarh
            )

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            idx (int): index in the dataset

        Returns:
            tuple: (image, target, idx)
        """
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)  # Channel is now first for Conv2d

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, idx

    def __len__(self) -> int:
        """Get the length of the dataset"""
        return len(self.data)

    def see_image(self, idx, show=True):
        """Get the image from the dataset and displays it.

        Args:
            idx (int): index of the image
            show (bool, optional): show the plot. Defaults to True.

        Returns:
            tuple: return the renderable image, with the correct
                label and incertainty labels.
        """
        true_label = self.true_targets[idx]
        img, label, _ = self.__getitem__(idx)  # img has channel as 1st dim
        img = np.transpose(img.numpy(), (1, 2, 0))  # channel as last dim
        if show:
            plt.imshow(img)
            plt.title(f"Label: {self.classes_labels[true_label]}")
            plt.show()
        else:
            return img, label, true_label

    @staticmethod
    def get_raw(path_data=path_data):
        """Return a pandas dataframe of the raw CIFAR10H data.
        `path_data` must contain the `cifar-10h-master folder."""
        folder = path.join(path_data, CIFAR10H.base_folder_cifarh, "data")
        file = path.join(
            folder,
            "cifar10h-raw.csv",
        )
        if not path.isfile(file):
            with zipfile.ZipFile(
                path.join(
                    folder,
                    "cifar10h-raw.zip",
                ),
                "r",
            ) as zip_ref:
                zip_ref.extractall(folder)
        df = pd.read_csv(file)
        return df
