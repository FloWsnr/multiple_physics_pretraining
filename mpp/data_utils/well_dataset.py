from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler


from einops import rearrange

from the_well.data.datasets import WellDataset
from the_well.data.normalization import ZScoreNormalization


class PhysicsDataset(WellDataset):
    """Wrapper around the WellDataset.

    Returns a dictionary with keys:
        - "pixel_values": input tensor of shape (c, h, w)
        - "labels": label tensor of shape (c, h, w)
        - "time": time value (currently set to 1)
        - "pixel_mask": mask tensor of shape (c,)

    Parameters
    ----------
    data_dir : Path
        Path to the data directory (e.g. "data/physics_data/train")
    use_normalization: bool
        Whether to use normalization
        By default False
    dt_stride: int
        Time step stride between samples
        By default 1
    full_trajectory_mode: bool
        Whether to use the full trajectory mode of the well dataset.
        This returns full trajectories instead of individual timesteps.
        By default False
    nan_to_zero: bool
        Whether to replace NaNs with 0
        By default True
    num_channels: int
        Number of channels in the data
        By default 5
    """

    def __init__(
        self,
        data_dir: Path,
        use_normalization: bool = True,
        T_in: int = 16,
        T_out: int = 1,
        dt_stride: int | list[int] = 1,
        full_trajectory_mode: bool = False,
        nan_to_zero: bool = True,
        num_channels: int = 5,
    ):
        if isinstance(dt_stride, list):
            min_dt_stride = dt_stride[0]
            max_dt_stride = dt_stride[1]
        else:
            min_dt_stride = dt_stride
            max_dt_stride = dt_stride

        super().__init__(
            path=str(data_dir),
            normalization_path=str(data_dir.parents[0] / "stats.yaml"),
            n_steps_input=T_in,
            n_steps_output=T_out,
            use_normalization=use_normalization,
            normalization_type=ZScoreNormalization,
            min_dt_stride=min_dt_stride,
            max_dt_stride=max_dt_stride,
            full_trajectory_mode=full_trajectory_mode,
        )
        self.nan_to_zero = nan_to_zero
        # give the dataset its correct name
        name = data_dir.parents[1].name
        self.dataset_name = name

        self.labels = torch.arange(num_channels)

    def __len__(self):
        return super().__len__()

    def __getitem__(
        self, index
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = super().__getitem__(index)  # returns (T, h, w, c)
        x = data["input_fields"]
        y = data["output_fields"]

        if self.nan_to_zero:
            x = torch.nan_to_num(x, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)
        # reshape to (t, c, h, w)
        x = rearrange(x, "t h w c -> t c h w")
        y = rearrange(y, "t h w c -> t c h w")

        bcs = data["boundary_conditions"]
        bc_x = bcs[0, 0]
        bc_y = bcs[1, 1]
        bcs = torch.tensor([bc_x, bc_y])

        return x, y, self.labels, bcs


class SuperDataset:
    """Wrapper around a list of datasets.

    Allows to concatenate multiple datasets and randomly sample from them.

    Parameters
    ----------
    datasets : dict[str, PhysicsDataset]
        Dictionary of datasets to concatenate

    max_samples_per_ds : Optional[int | list[int]]
        Maximum number of samples to sample from each dataset.
        If a list, specifies the number of samples for each dataset individually.
        If None, uses all samples from each dataset.
        By default None.


    seed : Optional[int]
        Random seed for reproducibility.
        By default None.
    """

    def __init__(
        self,
        datasets: dict[str, PhysicsDataset],
        max_samples_per_ds: Optional[int | list[int]] = None,
        seed: Optional[int] = None,
    ):
        self.datasets = datasets
        self.dataset_list = list(datasets.values())

        if isinstance(max_samples_per_ds, int):
            self.max_samples_per_ds = [max_samples_per_ds] * len(datasets)
        else:
            self.max_samples_per_ds = max_samples_per_ds

        self.seed = seed

        # Initialize random number generator
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        # Generate initial random indices
        self.reshuffle()

    def reshuffle(self):
        """Reshuffle the indices for each dataset.

        This should be called at the start of each epoch to ensure
        a new random subset of samples is used.

        """
        self.dataset_indices = []
        for i, dataset in enumerate(self.dataset_list):
            if (
                self.max_samples_per_ds is not None
                and len(dataset) > self.max_samples_per_ds[i]
            ):
                indices = torch.randperm(len(dataset), generator=self.rng)[
                    : self.max_samples_per_ds[i]
                ]
                self.dataset_indices.append(indices)
            else:
                self.dataset_indices.append(None)

        # Calculate lengths based on either max_samples_per_ds or full dataset length
        self.lengths = [
            min(self.max_samples_per_ds[i], len(dataset))
            if self.max_samples_per_ds is not None
            else len(dataset)
            for i, dataset in enumerate(self.dataset_list)
        ]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(
        self, index
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[int]]:
        for i, length in enumerate(self.lengths):
            if index < length:
                if self.dataset_indices[i] is not None:
                    # Use random index if available
                    actual_index = self.dataset_indices[i][index]
                else:
                    actual_index = index
                x, y, labels, bcs = self.dataset_list[i][
                    actual_index
                ]  # (time, h, w, n_channels)
                break
            index -= length
        return x, i, labels, bcs, y


def get_dataset(
    path: str,
    split_name: str,
    datasets: list,
    num_channels: int,
    min_stride: int = 1,
    max_stride: int = 1,
    T_in: int = 16,
    T_out: int = 1,
    use_normalization: bool = True,
    full_trajectory_mode: bool = False,
    nan_to_zero: bool = True,
) -> SuperDataset:
    """ """

    all_ds = {}
    for ds_name in datasets:
        ds_path = Path(path) / f"{ds_name}/data/{split_name}"
        if ds_path.exists():
            dataset = PhysicsDataset(
                data_dir=Path(path) / f"{ds_name}/data/{split_name}",
                use_normalization=use_normalization,
                T_in=T_in,
                T_out=T_out,
                dt_stride=[min_stride, max_stride],
                full_trajectory_mode=full_trajectory_mode,
                nan_to_zero=nan_to_zero,
                num_channels=num_channels,
            )
            all_ds[ds_name] = dataset

        else:
            print(f"Dataset path {ds_path} does not exist. Skipping.")

    return SuperDataset(all_ds)


def get_dataloader(
    path: str,
    num_channels: int,
    split: str,
    datasets: list[str],
    min_stride: int,
    T_in: int,
    T_out: int,
    max_stride: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    is_distributed: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    """Get a dataloader for the dataset.

    Parameters
    ----------
    data_config : dict
        Configuration for the datasets.
    seed : int
        Seed for the dataset.
    batch_size : int
        Batch size.
    num_workers : int
        Number of workers.
    prefetch_factor : int
        Prefetch factor.
    split : str
        Split to load ("train", "val", "test")
    data_fraction : float
        Fraction of the dataset to use.
    is_distributed : bool
        Whether to use distributed sampling
    shuffle : bool
        Whether to shuffle the dataset
    """
    super_dataset = get_dataset(
        path,
        split,
        datasets=datasets,
        num_channels=num_channels,
        min_stride=min_stride,
        max_stride=max_stride,
        T_in=T_in,
        T_out=T_out,
        use_normalization=True,
        full_trajectory_mode=False,
        nan_to_zero=True,
    )

    if is_distributed:
        sampler = DistributedSampler(super_dataset, seed=seed, shuffle=shuffle)
    else:
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler = RandomSampler(super_dataset, generator=generator)
        else:
            sampler = SequentialSampler(super_dataset)
    dataloader = DataLoader(
        dataset=super_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )

    return dataloader
