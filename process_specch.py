from datasets import DatasetDict
from .audio_augmentator.utils import get_speech_timestamps, collect_chunks, tensor_normalization, \
    signal_energy_noise_search
import torch
import numpy as np

noises_dataset_path = r'C:\Users\LimpWinter\Documents\Projects\audio_augmentator\noises_dataset'
noises_dataset = DatasetDict.load_from_disk(dataset_dict_path=noises_dataset_path)

speech = noises_dataset['speech_noises']
silero_vad_model_path = "silero_vad.jit"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(silero_vad_model_path,
                       map_location=device)


def preprocess_speech(
        speech_array: np.ndarray,
        vad_model
) -> torch.tensor:
    noise_to_mix_tensor = torch.from_numpy(np.float32(speech_array))
    noise_to_mix_tensor = tensor_normalization(noise_to_mix_tensor)

    speech_timestamps = get_speech_timestamps(
        input_audio=noise_to_mix_tensor,
        silero_vad_model=vad_model
    )
    if len(speech_timestamps) >= 1:
        noise_to_mix_tensor = collect_chunks(
            speech_timestamps,
            noise_to_mix_tensor
        )
    noise_to_mix_tensor = torch.unsqueeze(noise_to_mix_tensor, 0)
    return noise_to_mix_tensor


def preprocess_other(
        audio_array: np.ndarray
):
    noise_to_mix_array = signal_energy_noise_search(audio_array)
    noise_to_mix_array = np.float32(noise_to_mix_array)
    noise_to_mix_tensor = torch.unsqueeze(torch.from_numpy(noise_to_mix_array), 0)
    return noise_to_mix_tensor


def speech_mapper(example):
    example['audio']['array'] = preprocess_speech(example['audio']['array'], vad_model=model)
    return example


speech.map(speech_mapper, num_proc=4)
