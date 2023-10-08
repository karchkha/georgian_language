import collections
import torch

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import Trainer
from transformers.trainer import (
    SequentialDistributedSampler, 
    SequentialSampler,
    DistributedSamplerWithLoop
)
from transformers.trainer import is_datasets_available
from torch.utils.data import Dataset, DataLoader

class CommonVoiceTrainer(Trainer):

    def _get_train_sampler(self):
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None 
        
        if self.args.world_size <= 1:
            return RandomSampler(self.train_dataset)
        elif self.args.parallel_mode == ParallelMode.TPU and not self.args.dataloader_drop_last:
            # Use a loop for TPUs when drop_last is False to have all batches have the same size.
            return DistributedSamplerWithLoop(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
            )
        else:
            return DistributedSampler(
                self.train_dataset, num_replicas=self.args.world_size, rank=self.args.process_index
            )
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def _get_eval_sampler(self, eval_dataset):
        if self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# kjfkljdfs