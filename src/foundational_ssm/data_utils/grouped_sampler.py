import math
import logging
import re
from typing import Dict, Optional
from functools import cached_property
from collections import defaultdict

import torch
from temporaldata import Interval
from torch_brain.data.dataset import DatasetIndex


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