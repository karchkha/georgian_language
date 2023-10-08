import torchaudio
import librosa

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import os

from transformers import Wav2Vec2Processor


import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


class CommonVoiceDataset(Dataset):

    def __init__(self, dataset, processor, column_names=None, sep="\t", downsample = False):
        self.data = dataset
        self.processor = processor
        self.column_names = column_names
        self.downsample = downsample  ### this variable is for downsampling audio for ex. to phone qaulity

    def __len__(self):
        return len(self.data)


    def speech_file_to_array_fn(self, batch):
        # speech_array, sampling_rate = torchaudio.load(batch["path"])
        speech_array, sampling_rate = batch["audio"]['array'], batch["audio"]['sampling_rate']
        batch["speech"] = speech_array #[0].numpy()
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["sentence"]
        return batch

    
    def resample(self, batch):

      ### if downsample is given we resaple audio to that number and then resample again to 16,000
      if self.downsample:
        starting_sample_rate = self.downsample
        batch["speech"] = librosa.resample(np.asarray(batch["speech"]), orig_sr = 48000, target_sr = starting_sample_rate)
      else:
        starting_sample_rate = 48000

      batch["speech"] = librosa.resample(np.asarray(batch["speech"]), orig_sr = starting_sample_rate, target_sr = 16_000)
      batch["sampling_rate"] = 16_000
      return batch

    
    def prepare_dataset(self, batch, column_names=None):
        batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values[0].tolist()

        # with self.processor.as_target_processor():
        #     batch["labels"] = self.processor(batch["target_text"]).input_ids

        batch["labels"] = self.processor(text=batch["target_text"]).input_ids

        if column_names and isinstance(column_names, list):
            batch = {name: batch[name] for name in column_names}
        
        return batch


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch = self.data[idx].copy()
        # batch = batch.to_dict()
        batch = self.speech_file_to_array_fn(batch)
        batch = self.resample(batch)
        batch = self.prepare_dataset(batch, self.column_names)

        return batch 




@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch