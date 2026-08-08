import os
from typing import Dict, Any

import torchaudio
from datasets import Dataset, DatasetDict


class BaseNoiseDataset(Dataset):
    """
    Base dataset class for exact noise data which locates in folder
    with the following structure:
        noise_type/
            1.wav
            2.wav

    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.file_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.wav'):
                    self.file_paths.append(os.path.join(dirpath, filename))
        self.num_files = len(self.file_paths)
        self._info = self._create_dataset_info()

    @property
    def info(self):
        return self._info
    
    @info.setter
    def info(self, new_info: dict):
        self._info = new_info

    def __getitem__(self, item):
        file_path = self.file_paths[item]
        waveform, _ = torchaudio.load(file_path)
        return waveform

    def __len__(self):
        return self.num_files
    
    def _create_dataset_info(self) -> Dict[str, Any]:
        total_size = sum(os.path.getsize(fp) for fp in self.file_paths)
        
        sample_info = {}
        if self.file_paths:
            try:
                sample_waveform, sample_rate = torchaudio.load(self.file_paths[0])
                sample_info = {
                    'sample_rate': sample_rate,
                    'channels': sample_waveform.shape[0]
                }
            except Exception:
                sample_info = {'sample_rate': 'unknown', 'channels': 'unknown'}
        
        return {
            'description': f'Noise dataset from {self.root_dir}',
            'num_examples': self.num_files,
            'total_size_bytes': total_size,
            'root_directory': self.root_dir,
            'file_extensions': ['.wav'],
            'audio_info': sample_info
        }
    

def get_composed_dataset(root_dir: str) -> DatasetDict:
    """
    :param root_dir: root directory where located all noises folders with
    the following structure:
        root_dir/
            noise_type_1/
                1.wav
                2.wav
            noise_type_2/
                1.wav
                2.wav
    :return: DatasetDict which contains BaseNoiseDataset objects
    """
    datasets = {}
    for label in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, label)):
            datasets[label] = BaseNoiseDataset(os.path.join(root_dir, label))
    return DatasetDict(datasets)

