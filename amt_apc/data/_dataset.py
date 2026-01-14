from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from amt_apc.utils import config, info, get_package_root
from .sv.sampler import Sampler as SVSampler


def _get_dataset_dirs():
    """Get dataset directory paths."""
    root = get_package_root()
    dir_dataset = root / config.path.dataset / "dataset/"
    dir_spec = dir_dataset / "spec/"
    dir_label = dir_dataset / "label/"
    return dir_spec, dir_label


class PianoCoversDataset(Dataset):
    def __init__(self, split="train"):
        _, dir_label = _get_dataset_dirs()
        self.data = list(dir_label.glob("*.npz"))
        if split == "train":
            self.data = [path for path in self.data if self.is_train(path)]
        elif split == "test":
            self.data = [path for path in self.data if not self.is_train(path)]
        elif split == "all":
            pass
        else:
            raise ValueError(f"Invalid value for 'split': {split}")
        self.sv_sampler = SVSampler()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = np.load(path)
        spec, sv = self.get_spec_sv(path)

        spec = torch.from_numpy(spec).float()
        sv = torch.tensor(sv).float()
        onset = torch.from_numpy(label["onset"])
        offset = torch.from_numpy(label["offset"])
        frame = torch.from_numpy(label["frame"])
        velocity = torch.from_numpy(label["velocity"]).long()

        return spec, sv, onset, offset, frame, velocity

    @staticmethod
    def get_id_n(path: Path):
        split = path.stem.split("_")
        n_segment = split[-1]
        id_piano = "_".join(split[:-1])
        return id_piano, n_segment

    def is_train(self, path: Path):
        return info.is_train(self.get_id_n(path)[0])

    def get_spec_sv(self, path: Path):
        dir_spec, _ = _get_dataset_dirs()
        id_piano, n_segment = self.get_id_n(path)
        id_orig = info.piano2orig(id_piano)
        fname_orig = f"{id_orig}_{n_segment}.npy"
        path_orig = dir_spec / fname_orig
        spec = np.load(path_orig)
        sv = self.sv_sampler[id_piano]
        return spec, sv
