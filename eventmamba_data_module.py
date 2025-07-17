from typing import Any, Dict, Optional
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import os
import sys
import h5py

def load_h5_mark(h5_filename):
    print(h5_filename)
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    mark = f['mark'][:]
    return (data, label,mark)

def load_h5(h5_filename):
    print(h5_filename)
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_h5_and_resample(h5_filename, sample_size=1024):
    print(f"Loading {h5_filename}")
    with h5py.File(h5_filename, 'r') as f:
        data = []
        labels = []
        for group_name in f:
            group = f[group_name]
            x = group['x'][:]
            y = group['y'][:]
            t = group['t'][:]
            current_sample_size = len(x)
            if current_sample_size < sample_size:
                repeat_factor = np.ceil(sample_size / current_sample_size).astype(int)
                x = np.tile(x, repeat_factor)[:sample_size]
                y = np.tile(y, repeat_factor)[:sample_size]
                t = np.tile(t, repeat_factor)[:sample_size]
            else:
                indices = np.random.choice(current_sample_size, sample_size, replace=False)
                indices = np.argsort(indices)
                x = x[indices]
                y = y[indices]
                t = t[indices]
            if current_sample_size >= 1024:
                sample_data = np.stack((t,x,y), axis=-1)
                data.append(sample_data)
                sample_label = group.attrs['label']
                labels.append(sample_label)
    return (data, labels)



class EventMambaDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/DVS-Lip/",
        batch_size: int = 128,
        num_workers: int = 4,
        num_point: int = 2048,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_point = num_point
        self.data_train: Optional[Dataset] = None
        self.data_val_test: Optional[Dataset] = None
        self._empty_dataset = TensorDataset(torch.empty(0, self.num_point, 3), torch.empty(0, dtype=torch.long))

    def prepare_data(self) -> None:
        # 假设h5文件已存在，无需下载
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is None:
            train_file = self.data_dir + "train.h5"
            current_data_train, current_label_train = load_h5(train_file)
            current_label_train = np.array(current_label_train)
            current_data_train = torch.from_numpy(current_data_train).reshape(-1, self.num_point, 3)
            current_label_train = torch.from_numpy(current_label_train.astype('int64'))
            self.data_train = TensorDataset(current_data_train, current_label_train)
        if self.data_val_test is None:
            test_file = self.data_dir + "test.h5"
            current_data_test, current_label_test = load_h5(test_file)
            current_label_test = np.array(current_label_test)
            current_data_test = torch.from_numpy(current_data_test).reshape(-1, self.num_point, 3)
            current_label_test = torch.from_numpy(current_label_test.astype('int64'))
            self.data_val_test = TensorDataset(current_data_test, current_label_test)

    def train_dataloader(self) -> DataLoader[Any]:
        dataset = self.data_train if self.data_train is not None else self._empty_dataset
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        dataset = self.data_val_test if self.data_val_test is not None else self._empty_dataset
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        dataset = self.data_val_test if self.data_val_test is not None else self._empty_dataset
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

if __name__ == "__main__":
    dm = EventMambaDataModule()
    dm.prepare_data()
    dm.setup()
    def get_len(ds):
        if ds is None:
            return 0
        if hasattr(ds, 'tensors'):
            return len(ds.tensors[0])
        if hasattr(ds, 'datasets'):
            return sum(get_len(d) for d in ds.datasets)
        return len(ds)
    print("train: ", get_len(dm.data_train))
    print("val/test: ", get_len(dm.data_val_test)) 