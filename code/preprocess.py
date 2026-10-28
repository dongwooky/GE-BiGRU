import numpy as np
from tqdm import trange
import dateutil.parser
from datetime import timedelta
import torch
from torch.utils.data import Dataset

class HaiDataset(Dataset):
    def __init__(self, timestamps, window_size, df, stride=1, attacks=None):
        self.ts = np.array(timestamps)
        self.window_size = window_size
        self.tag_values = np.array(df, dtype=np.float32)
        self.stride = stride

        self.valid_idxs = [
            L for L in trange(len(self.ts) - self.window_size + 1)
            if dateutil.parser.parse(self.ts[L + self.window_size - 1]) - dateutil.parser.parse(self.ts[L]) == timedelta(seconds=self.window_size - 1)
        ]
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::self.stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}")

        self.with_attack = attacks is not None
        if self.with_attack:
            self.attacks = np.array(attacks, dtype=np.float32)

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + self.window_size - 1
        item = {"ts": self.ts[last]}
        if self.with_attack:
            item["attack"] = self.attacks[last]
        item["given"] = torch.from_numpy(self.tag_values[i:last])
        item["answer"] = torch.from_numpy(self.tag_values[last])
        return item
