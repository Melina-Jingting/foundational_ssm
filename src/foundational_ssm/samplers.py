import math
import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from functools import cached_property
from collections import defaultdict

import torch
from temporaldata import Interval
from torch_brain.data.dataset import DatasetIndex
import numpy as np
import time

@dataclass
class DatasetIndex:
    r"""The dataset can be indexed by specifying a recording id and a start and end time."""

    recording_id: str
    start: float
    end: float
    
class RandomFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows randomly, given intervals defined in the
    :obj:`sampling_intervals` parameter. :obj:`sampling_intervals` is a dictionary where the keys
    are the session ids and the values are lists of tuples representing the
    start and end of the intervals from which to sample. The samples are shuffled, and
    random temporal jitter is applied.


    In one epoch, the number of samples that is generated from a given sampling interval
    is given by:

    .. math::
        N = \left\lfloor\frac{\text{interval_length}}{\text{window_length}}\right\rfloor

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
        drop_short (bool, optional): Whether to drop short intervals. Defaults to True.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        generator: Optional[torch.Generator] = None,
        drop_short: bool = True,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.generator = generator
        self.drop_short = drop_short

    @cached_property
    def _estimated_len(self):
        num_samples = 0
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                num_samples += math.floor(interval_length / self.window_length)

        if self.drop_short and total_short_dropped > 0:
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")
        return num_samples

    def __len__(self):
        return self._estimated_len

    def __iter__(self):
        if len(self) == 0.0:
            raise ValueError("All intervals are too short to sample from.")
        indices = []
        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                # sample a random offset
                left_offset = (
                    torch.rand(1, generator=self.generator).item() * self.window_length
                )

                indices_ = [
                    DatasetIndex(
                        session_name, t.item(), (t + self.window_length).item()
                    )
                    for t in torch.arange(
                        start + left_offset,
                        end,
                        self.window_length,
                        dtype=torch.float64,
                    )
                    if t + self.window_length <= end
                ]

                if len(indices_) > 0:
                    indices.extend(indices_)
                    right_offset = end - indices[-1].end
                else:
                    right_offset = end - start - left_offset

                # if there is one sample worth of data, add it
                # this ensures that the number of samples is always consistent
                if right_offset + left_offset >= self.window_length:
                    if right_offset > left_offset:
                        indices.append(
                            DatasetIndex(session_name, end - self.window_length, end)
                        )
                    else:
                        indices.append(
                            DatasetIndex(
                                session_name, start, start + self.window_length
                            )
                        )
        # shuffle
        for idx in torch.randperm(len(indices), generator=self.generator):
            yield indices[idx]
            

class SequentialFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows sequentially, always in the same order. The
    sampling intervals are defined in the :obj:`sampling_intervals` parameter.
    :obj:`sampling_intervals` is a dictionary where the keys are the session ids and the
    values are lists of tuples representing the start and end of the intervals
    from which to sample.

    If the length of a sequence is not evenly divisible by the step, the last
    window will be added with an overlap with the previous window. This is to ensure
    that the entire sequence is covered.

    Args:
        sampling_intervals (Dict[str, List[Tuple[float, float]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        step (float, optional): Step size between windows. If None, it
            defaults to ``window_length``.
        drop_short (bool, optional): Whether to drop windows smaller than ``window_length``.
            Defaults to False.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, List[Tuple[float, float]]],
        window_length: float,
        step: Optional[float] = None,
        drop_short=False,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.step = step or window_length
        self.drop_short = drop_short

        assert self.step > 0, "Step must be greater than 0."

    # we cache the indices since they are deterministic
    @cached_property
    def _indices(self) -> List[DatasetIndex]:
        indices = []
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                indices_ = [
                    DatasetIndex(
                        session_name, t.item(), (t + self.window_length).item()
                    )
                    for t in torch.arange(start, end, self.step, dtype=torch.float64)
                    if t + self.window_length <= end
                ]

                indices.extend(indices_)

                # we need to make sure that the entire interval is covered
                if indices_[-1].end < end:
                    indices.append(
                        DatasetIndex(session_name, end - self.window_length, end)
                    )

        if self.drop_short and total_short_dropped > 0:
            num_samples = len(indices)
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")

        return indices

    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        yield from self._indices


class TrialSampler(torch.utils.data.Sampler):
    r"""Randomly samples a single trial interval from the given intervals.

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
        shuffle (bool, optional): Whether to shuffle the indices. Defaults to False.
    """

    def __init__(
        self,
        *,
        
        sampling_intervals: Dict[str, List[Tuple[float, float]]],
        generator: Optional[torch.Generator] = None,
        shuffle: bool = False,
        min_interval_length: float = 0.5,
    ):
        self.sampling_intervals = sampling_intervals
        self.generator = generator
        self.shuffle = shuffle
        self.min_interval_length = min_interval_length

    def __len__(self):
        return sum(len(intervals) for intervals in self.sampling_intervals.values())

    def __iter__(self):
        # Flatten the intervals from all sessions into a single list
        all_intervals = [
            (session_id, start, end)
            for session_id, intervals in self.sampling_intervals.items()
            for start, end in zip(intervals.start, intervals.end)
            if end - start >= self.min_interval_length
        ]

        indices = [
            DatasetIndex(session_id, start, end)
            for session_id, start, end in all_intervals
        ]

        if self.shuffle:
            # Yield a single DatasetIndex representing the selected interval
            for idx in torch.randperm(len(indices), generator=self.generator):
                yield indices[idx]
        else:
            yield from indices

class GroupedRandomFixedWindowSampler(torch.utils.data.Sampler):
    """
    Samples fixed-length windows randomly, but only from within one dataset/subject/task 
    combination at a time. Groups sessions by the pattern {dataset}/{subject}_{date}_{task}
    and samples from one group per batch.
    
    Args:
        sampling_intervals (Dict[str, Interval]): Sampling intervals for each session.
        window_length (float): Length of the window to sample.
        batch_size (int): Number of samples per batch.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
        drop_short (bool, optional): Whether to drop short intervals. Defaults to True.
    """
    
    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        drop_short: bool = True,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.batch_size = batch_size
        self.generator = generator
        self.drop_short = drop_short
        
        # Group sessions by dataset/subject/task
        self.session_groups = self._group_sessions()
        
    def _group_sessions(self):
        """Group sessions by dataset/subject/task combination."""
        groups = defaultdict(list)
        
        for session_name in self.sampling_intervals.keys():
            # Parse session name: {dataset}/{subject}_{date}_{task}
            # Example: "perich_miller_population_2018/c_20131003_center_out_reaching"
            match = re.match(r'([^/]+)/([^_]+)_[^_]+_(.+)', session_name)
            if match:
                dataset, subject, task = match.groups()
                group_key = f"{dataset}/{subject}_{task}"
                groups[group_key].append(session_name)
            else:
                # Fallback: use session name as group if parsing fails
                groups[session_name].append(session_name)
        
        return dict(groups)
    
    @cached_property
    def _estimated_len(self):
        """Estimate total number of samples across all groups."""
        num_samples = 0
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                num_samples += math.floor(interval_length / self.window_length)

        if self.drop_short and total_short_dropped > 0:
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")
        return num_samples

    def __len__(self):
        return self._estimated_len // self.batch_size

    def __iter__(self):
        if len(self) == 0.0:
            raise ValueError("All intervals are too short to sample from.")

        # Get all group keys and shuffle them
        group_keys = list(self.session_groups.keys())
        if self.generator is not None:
            perm = torch.randperm(len(group_keys), generator=self.generator)
            group_keys = [group_keys[i] for i in perm]
        else:
            import random
            random.shuffle(group_keys)

        # Collect all batches from all groups
        all_batches = []
        for group_key in group_keys:
            group_sessions = self.session_groups[group_key]
            group_indices = []
            for session_name in group_sessions:
                sampling_intervals = self.sampling_intervals[session_name]
                for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                    interval_length = end - start
                    if interval_length < self.window_length:
                        if self.drop_short:
                            continue
                        else:
                            raise ValueError(
                                f"Interval {(start, end)} is too short to sample from. "
                                f"Minimum length is {self.window_length}."
                            )
                    left_offset = (
                        torch.rand(1, generator=self.generator).item() * self.window_length
                    )
                    indices_ = [
                        DatasetIndex(
                            session_name, t.item(), (t + self.window_length).item()
                        )
                        for t in torch.arange(
                            start + left_offset,
                            end,
                            self.window_length,
                            dtype=torch.float64,
                        )
                        if t + self.window_length <= end
                    ]
                    if len(indices_) > 0:
                        group_indices.extend(indices_)
                        right_offset = end - indices_[-1].end
                    else:
                        right_offset = end - start - left_offset
                    if right_offset + left_offset >= self.window_length:
                        if right_offset > left_offset:
                            group_indices.append(
                                DatasetIndex(session_name, end - self.window_length, end)
                            )
                        else:
                            group_indices.append(
                                DatasetIndex(
                                    session_name, start, start + self.window_length
                                )
                            )
            # Shuffle samples within this group
            if self.generator is not None:
                perm = torch.randperm(len(group_indices), generator=self.generator)
                group_indices = [group_indices[i] for i in perm]
            else:
                import random
                random.shuffle(group_indices)
            # Create batches for this group
            for i in range(0, len(group_indices), self.batch_size):
                batch = group_indices[i:i + self.batch_size]
                all_batches.append(batch)

        # Shuffle all batches together
        if self.generator is not None:
            perm = torch.randperm(len(all_batches), generator=self.generator)
            all_batches = [all_batches[i] for i in perm]
        else:
            import random
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch
            
class GroupedSequentialFixedWindowSampler(torch.utils.data.Sampler):
    """
    Samples fixed-length windows randomly, but only from within one dataset/subject/task 
    combination at a time. Groups sessions by the pattern {dataset}/{subject}_{date}_{task}
    and samples from one group per batch.
    
    Args:
        sampling_intervals (Dict[str, Interval]): Sampling intervals for each session.
        window_length (float): Length of the window to sample.
        batch_size (int): Number of samples per batch.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
        drop_short (bool, optional): Whether to drop short intervals. Defaults to True.
    """
    
    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        drop_short: bool = True,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.batch_size = batch_size
        self.generator = generator
        self.drop_short = drop_short
        
        # Group sessions by dataset/subject/task
        self.session_groups = self._group_sessions()
        
    def _group_sessions(self):
        """Group sessions by dataset/subject/task combination."""
        groups = defaultdict(list)
        
        for session_name in self.sampling_intervals.keys():
            # Parse session name: {dataset}/{subject}_{date}_{task}
            # Example: "perich_miller_population_2018/c_20131003_center_out_reaching"
            match = re.match(r'([^/]+)/([^_]+)_[^_]+_(.+)', session_name)
            if match:
                dataset, subject, task = match.groups()
                group_key = f"{dataset}/{subject}_{task}"
                groups[group_key].append(session_name)
            else:
                # Fallback: use session name as group if parsing fails
                groups[session_name].append(session_name)
        
        return dict(groups)
    
    @cached_property
    def _estimated_len(self):
        """Estimate total number of samples across all groups."""
        num_samples = 0
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                num_samples += math.floor(interval_length / self.window_length)

        if self.drop_short and total_short_dropped > 0:
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")
        return num_samples

    def __len__(self):
        return self._estimated_len // self.batch_size

    def __iter__(self):
        if len(self) == 0.0:
            raise ValueError("All intervals are too short to sample from.")

        # Get all group keys and sort them
        group_keys = list(self.session_groups.keys())
        group_keys.sort()

        # Collect all batches from all groups
        all_batches = []
        for group_key in group_keys:
            group_sessions = self.session_groups[group_key]
            group_indices = []
            for session_name in group_sessions:
                sampling_intervals = self.sampling_intervals[session_name]
                for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                    interval_length = end - start
                    if interval_length < self.window_length:
                        if self.drop_short:
                            continue
                        else:
                            raise ValueError(
                                f"Interval {(start, end)} is too short to sample from. "
                                f"Minimum length is {self.window_length}."
                            )
                    left_offset = (
                        torch.rand(1, generator=self.generator).item() * self.window_length
                    )
                    indices_ = [
                        DatasetIndex(
                            session_name, t.item(), (t + self.window_length).item()
                        )
                        for t in torch.arange(
                            start + left_offset,
                            end,
                            self.window_length,
                            dtype=torch.float64,
                        )
                        if t + self.window_length <= end
                    ]
                    if len(indices_) > 0:
                        group_indices.extend(indices_)
                        right_offset = end - indices_[-1].end
                    else:
                        right_offset = end - start - left_offset
                    if right_offset + left_offset >= self.window_length:
                        if right_offset > left_offset:
                            group_indices.append(
                                DatasetIndex(session_name, end - self.window_length, end)
                            )
                        else:
                            group_indices.append(
                                DatasetIndex(
                                    session_name, start, start + self.window_length
                                )
                            )
            # Create batches for this group
            for i in range(0, len(group_indices), self.batch_size):
                batch = group_indices[i:i + self.batch_size]
                all_batches.append(batch)


        for batch in all_batches:
            yield batch
            
            
        
