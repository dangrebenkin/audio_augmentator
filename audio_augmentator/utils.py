import numpy as np
import struct
from scipy import signal
import torch
from typing import Callable, List
import warnings
import os
import torchaudio
from torch.utils.data import Dataset
from datasets import DatasetDict


def signal_energy_noise_search(
        audio_noise_numpy_array: np.ndarray,
        sample_rate: int = 16_000
) -> np.ndarray:
    sample_rate = sample_rate
    interval = int(3.0 * sample_rate)
    nperseg = int(sample_rate / 100)

    input_audio_bytes = np.asarray(audio_noise_numpy_array * 32768.0, dtype=np.int16).tobytes()
    n_data = len(input_audio_bytes)
    sound_signal = np.empty((int(n_data / 2),))

    for ind in range(sound_signal.shape[0]):
        sound_signal[ind] = float(struct.unpack('<h', input_audio_bytes[(ind * 2):(ind * 2 + 2)])[0])

    frequencies_axis, time_axis, spectrogram = signal.spectrogram(
        sound_signal,
        fs=sample_rate,
        window='hamming',
        nperseg=nperseg,
        noverlap=0,
        scaling='spectrum',
        mode='psd'
    )
    frame_size = int(round(0.001 * float(sample_rate)))
    spectrogram = spectrogram.transpose()
    sound_frames = np.reshape(sound_signal[0:(spectrogram.shape[0] * frame_size)],
                              (spectrogram.shape[0], frame_size))
    # window energy
    energy_values = []
    for time_ind in range(spectrogram.shape[0]):
        energy = np.square(sound_frames[time_ind]).mean()
        energy_values.append(energy)

    # local minimums search
    energy_minimums_indices = []
    for i in range(len(energy_values) - 1):
        if (energy_values[i] < energy_values[i - 1]) and (energy_values[i] < energy_values[i + 1]):
            energy_minimums_indices.append(i)
    energy_minimums_indices.append(len(energy_values) - 1)
    minimums = [i * nperseg for i in energy_minimums_indices]
    if minimums[0] != 0:
        minimums.insert(0, 0)

    # local maximums search
    energy_maximums_indices = []
    energy_maximums_values = []
    for i in range(len(energy_values) - 1):
        if (energy_values[i] > energy_values[i - 1]) and (energy_values[i] > energy_values[i + 1]):
            energy_maximums_indices.append(i)
            energy_maximums_values.append(energy_values[i])

    max_maximum_index = 0
    if len(energy_maximums_indices) > 1:
        max_maximum = max(energy_maximums_values)
        max_maximum_index = energy_maximums_values.index(max_maximum)
        max_maximum_index = energy_maximums_indices[max_maximum_index]
    elif len(energy_maximums_indices) == 1:
        max_maximum_index = energy_maximums_indices[0]
    max_maximum_index = max_maximum_index * nperseg

    upper_time_bound = max_maximum_index + interval
    lower_time_bound = max_maximum_index - interval
    start_minimums = [i for i in minimums if (i >= lower_time_bound) and (i < max_maximum_index)]
    finish_minimums = [i for i in minimums if (i <= upper_time_bound) and (i > max_maximum_index)]
    startpoint = min(start_minimums)
    finishpoint = max(finish_minimums)
    noise_fragment = audio_noise_numpy_array[startpoint:finishpoint]
    return noise_fragment


def get_speech_timestamps(
        input_audio: torch.Tensor,
        silero_vad_model,
        threshold: float = 0.5,
        sampling_rate_value: int = 16000,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float('inf'),
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30,
        return_seconds: bool = False,
        progress_tracking_callback: Callable[[float], None] = None
):
    if not torch.is_tensor(input_audio):
        try:
            input_audio = torch.Tensor(input_audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(input_audio.shape) > 1:
        for i in range(len(input_audio.shape)):  # trying to squeeze empty dimensions
            input_audio = input_audio.squeeze(0)
        if len(input_audio.shape) > 1:
            raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

    if sampling_rate_value > 16000 and (sampling_rate_value % 16000 == 0):
        step = sampling_rate_value // 16000
        sampling_rate_value = 16000
        input_audio = input_audio[::step]
        warnings.warn('Sampling rate is a multiply of 16000, casting to 16000 manually!')
    else:
        step = 1

    if sampling_rate_value == 8000 and window_size_samples > 768:
        warnings.warn(
            'window_size_samples is too big for 8000 sampling_rate! Better set window_size_samples to 256, '
            '512 or 768 for 8000 sample rate!')
    if window_size_samples not in [256, 512, 768, 1024, 1536]:
        warnings.warn(
            'Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 '
            'sampling_rate\n - [256, 512, 768] for 8000 sampling_rate')

    silero_vad_model.reset_states()
    min_speech_samples = sampling_rate_value * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate_value * speech_pad_ms / 1000
    max_speech_samples = sampling_rate_value * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate_value * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate_value * 98 / 1000

    audio_length_samples = len(input_audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = input_audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob = silero_vad_model(chunk, sampling_rate_value).item()
        speech_probs.append(speech_prob)
        # caculate progress and seng it to callback function
        progress = current_start_sample + window_size_samples
        if progress > audio_length_samples:
            progress = audio_length_samples
        progress_percent = (progress / audio_length_samples) * 100
        if progress_tracking_callback:
            progress_tracking_callback(progress_percent)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0  # to save potential segment end (and tolerate some silence)
    prev_end = next_start = 0  # to save potential segment limits in case of maximum segment size reached

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = window_size_samples * i
            continue

        if triggered and (window_size_samples * i) - current_speech['start'] > max_speech_samples:
            if prev_end:
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end:  # previously reached silence (< neg_thres)
                    # and is still not speech (< thres)
                    triggered = False
                else:
                    current_speech['start'] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech['end'] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if ((
                        window_size_samples * i) - temp_end) > min_silence_samples_at_max_speech:
                # condition to avoid cutting in very short silence
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i + 1]['start'] = int(max(0, speeches[i + 1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i + 1]['start'] = int(max(0, speeches[i + 1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    if return_seconds:
        for speech_dict in speeches:
            speech_dict['start'] = round(speech_dict['start'] / sampling_rate_value, 1)
            speech_dict['end'] = round(speech_dict['end'] / sampling_rate_value, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict['start'] *= step
            speech_dict['end'] *= step

    return speeches


# collect_chunks and get_speech_timestamps from https://github.com/snakers4/silero-vad/blob/master/utils_vad.p
def collect_chunks(
        tss: List[dict],
        wav: torch.Tensor
):
    chunks = []
    for i in tss:
        chunks.append(wav[i['start']: i['end']])
    return torch.cat(chunks)


def tensor_normalization(input_tensor: torch.tensor) -> torch.tensor:
    if torch.max(torch.abs(input_tensor)).item() > 1.0:
        input_tensor /= torch.max(torch.abs(input_tensor))
    return input_tensor


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


def get_composed_dataset(root_dir: str):
    datasets = {}
    for label in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, label)):
            datasets[label] = BaseNoiseDataset(os.path.join(root_dir, label))
    return DatasetDict(datasets)


class BaseNoiseDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.file_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.wav'):
                    self.file_paths.append(os.path.join(dirpath, filename))
        self.num_files = len(self.file_paths)

    def __getitem__(self, item):
        file_path = self.file_paths[item]
        waveform, _ = torchaudio.load(file_path)
        return waveform

    def __len__(self):
        return self.num_files
