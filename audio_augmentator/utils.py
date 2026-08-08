import os
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import numpy.typing as npt
import torch
from huggingface_hub import snapshot_download
from scipy import signal
from silero_vad import get_speech_timestamps
from tqdm import tqdm


def audio_duration(audio_array: npt.NDArray, 
                   sample_rate: int = 16000) -> float:
    num_samples = audio_array.shape[0]
    duration = num_samples / sample_rate
    return duration


def signal_energy_noise_search(
        audio_noise_numpy_array: npt.NDArray,
        sample_rate: int = 16_000) -> np.ndarray:

    interval = int(3 * sample_rate)
    nperseg = int(sample_rate / 100)

    _, _, spectrogram = signal.spectrogram(
        audio_noise_numpy_array,
        fs=sample_rate,
        window='hamming',
        nperseg=nperseg,
        noverlap=0,
        scaling='spectrum',
        mode='psd'
    )
    spectrogram = spectrogram.transpose()

    # window energy
    energy_values = []
    for time_ind in range(spectrogram.shape[0]):
        energy = np.square(spectrogram[time_ind]).mean()
        energy_values.append(energy)
    energy_indices = [(i, idx) for idx, i in enumerate(energy_values)]
    
    # local minimums search
    energy_minimums_indices = []
    for i in energy_indices[1:-2]:
        value, index = i
        if (value < energy_values[index - 1]) and (value < energy_values[index + 1]):
            energy_minimums_indices.append(index)
    energy_minimums_indices.append(energy_indices[-1][1])
    minimums = [i * nperseg for i in energy_minimums_indices]
    if minimums[0] != 0:
        minimums.insert(0, 0)

    # global maximum index
    global_maximum_index = max(energy_indices, key=lambda x: x[0])[1]
    global_maximum_index = global_maximum_index * nperseg

    upper_time_bound = global_maximum_index + interval
    lower_time_bound = global_maximum_index - interval
    start_minimums = [i for i in minimums if (i >= lower_time_bound) and (i < global_maximum_index)]
    finish_minimums = [i for i in minimums if (i <= upper_time_bound) and (i > global_maximum_index)]
    startpoint = min(start_minimums)
    finishpoint = max(finish_minimums)
    noise_fragment = audio_noise_numpy_array[startpoint:finishpoint]
    return noise_fragment


# collect_chunks and get_speech_timestamps from https://github.com/snakers4/silero-vad/blob/master/utils_vad.p
def collect_chunks(
        tss: List[dict],
        wav: torch.Tensor
):
    chunks = []
    for i in tss:
        chunks.append(wav[i['start']: i['end']])
    return torch.cat(chunks)


def tensor_normalization(input_tensor: torch.Tensor) -> torch.Tensor:
    if torch.max(torch.abs(input_tensor)).item() > 1.0:
        input_tensor /= torch.max(torch.abs(input_tensor))
    return input_tensor


def preprocess_speech(
        speech_array: npt.NDArray,
        vad_model,
        sample_rate: int = 16000
    ) -> torch.Tensor:

    noise_to_mix_tensor = torch.from_numpy(np.float32(speech_array))
    noise_to_mix_tensor = tensor_normalization(noise_to_mix_tensor)

    if audio_duration(speech_array, sample_rate) <= 3.0:
        return noise_to_mix_tensor

    speech_timestamps = get_speech_timestamps(
        noise_to_mix_tensor,
        vad_model,
        sample_rate
    )
    if len(speech_timestamps) >= 1:
        noise_to_mix_tensor = collect_chunks(
            speech_timestamps,
            noise_to_mix_tensor
        )
    noise_to_mix_tensor = torch.unsqueeze(noise_to_mix_tensor, 0)
    return noise_to_mix_tensor


def preprocess_noise(
        audio_array: npt.NDArray,
        sample_rate: int = 16000
    ) -> torch.Tensor:
    noise_to_mix_tensor = torch.from_numpy(np.float32(audio_array))
    noise_to_mix_tensor = tensor_normalization(noise_to_mix_tensor)
    if noise_to_mix_tensor.size()[0] != 1:
        noise_to_mix_tensor = torch.unsqueeze(noise_to_mix_tensor, 0)
    if audio_duration(audio_array, sample_rate) <= 3.0:
        return noise_to_mix_tensor
    noise_to_mix_array = signal_energy_noise_search(audio_array, sample_rate)
    noise_to_mix_array = np.float32(noise_to_mix_array)
    noise_to_mix_tensor_final = torch.from_numpy(noise_to_mix_array)
    if noise_to_mix_tensor_final.size()[0] != 1:
        noise_to_mix_tensor_final = torch.unsqueeze(noise_to_mix_tensor_final, 0)
    if noise_to_mix_tensor_final.numel() == 0:
        return noise_to_mix_tensor
    return noise_to_mix_tensor_final


def measure_time_for_augmentation(
        augmentator,
        data: Iterable,
        sample_rate: int = 16000
    ):
    durations: list[float] = []
    times_for_augmentations: list[float] = []
    augmented_audios: list[torch.Tensor] = []
    for sample in tqdm(data):
        audio_array = sample["audio"]["array"]
        audio_duration = audio_array.size / sample_rate
        durations.append(audio_duration)
        audio_tensor = torch.unsqueeze(torch.from_numpy(audio_array), 0)

        start_time = time.time()
        augmented_audio = augmentator.run(audio_input=audio_tensor, augment=True)
        end_time = time.time()
        
        augmented_audios.append(augmented_audio)

        for _, aug_sample in augmented_audio.items():
            assert aug_sample.shape == audio_tensor.shape, \
                f"Shapes are not equal! Expected shape: {audio_tensor.shape}, got shape: {aug_sample.shape}"
        times_for_augmentations.append(float(end_time - start_time))
    return durations, times_for_augmentations, augmented_audios


def download_dataset(dataset_name, revision=None, cache_dir=None):
    if cache_dir:
        local_dir = Path(cache_dir) / Path(dataset_name).name
    else:
        local_dir = f"./datasets/{Path(dataset_name).name}"
    
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading dataset {dataset_name}...")
    if revision:
        print(f"Revision: {revision}")
    
    try:
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=local_dir,
            revision=revision,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"Dataset successfully downloaded to {local_dir}")
        return local_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None



