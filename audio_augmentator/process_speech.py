import argparse
import os
from pathlib import Path

import torch
from silero_vad import load_silero_vad
from datasets import DatasetDict, load_from_disk

from .utils import preprocess_other, preprocess_speech


def speech_mapper(example):
    example['audio']['array'] = preprocess_speech(example['audio']['array'], 
                                                  vad_model=model, 
                                                  sample_rate=sample_rate)
    return example

def nonspeech_noise_mapper(example):
    example['audio']['array'] = preprocess_other(example['audio']['array'],
                                                 sample_rate=sample_rate)
    return example


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir',
                        dest='dataset_dir', type=str, required=True)
    parser.add_argument('-s', '--dataset_split',
                        dest='split', default=None, type=str, required=False)
    parser.add_argument('-sr', '--sample_rate',
                        dest='sample_rate', type=str, required=True)
    parser.add_argument('-v', '--vad',
                        dest='use_vad', action="store_true", required=False)
    args = parser.parse_args()
    
    sample_rate = args.sample_rate
    split = args.split

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        raise ValueError(f'Directory {args.dataset_dir} does not exist!')
    

    loaded_dataset = load_from_disk(dataset_path)
    if split is not None and isinstance(loaded_dataset, DatasetDict):
        loaded_dataset = loaded_dataset[split]
    
    if args.use_vad:
        if not Path(args.vad_model).exists():
            raise ValueError(f'Path {args.vad_model} does not exist!')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_silero_vad()
        loaded_dataset_processed = loaded_dataset.map(speech_mapper,
                                                      batched=False,
                                                      num_proc=os.cpu_count())
    else:
        loaded_dataset_processed = loaded_dataset.map(speech_mapper,
                                                      batched=False,
                                                      num_proc=os.cpu_count())
    if split is not None:
        loaded_dataset[split] = loaded_dataset_processed
    
    loaded_dataset.save_to_disk(f'processed_{dataset_path.name}')

