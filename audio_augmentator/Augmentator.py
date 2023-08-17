import math
import torch
import random
import struct
import string
import warnings
import torchaudio
import numpy as np
from pathlib import Path
from scipy import signal
from datasets import DatasetDict
from typing import Callable, List
from pysndfx import AudioEffectsChain


class Augmentator:

    def __init__(self,

                 noises_dataset: str,
                 silero_vad_model_path: str = None,
                 decibels: float = 10.0,
                 household_noises: bool = False,
                 pets_noises: bool = False,
                 speech_noises: bool = False,
                 background_music_noises: bool = False,
                 to_mix: bool = False,

                 to_reverb: bool = False,
                 reverberance: int = 50,
                 hf_damping: int = 50,
                 room_scale: int = 100,
                 stereo_depth: int = 100,
                 pre_delay: int = 20,
                 wet_gain: int = 0,
                 wet_only: bool = False
                 ):

        self.device = torch.device("cpu")
        self.model = None
        if silero_vad_model_path is not None:
            model = torch.jit.load(silero_vad_model_path,
                                   map_location=self.device)
            self.model = model.to(self.device)
        self.noises_dataset = DatasetDict.load_from_disk(noises_dataset)
        self.resampler = torchaudio.transforms.Resample(new_freq=16000).to(self.device)
        self.overlayer = torchaudio.transforms.AddNoise().to(self.device)
        self.sample_rate = 16000
        self.nperseg = int(self.sample_rate / 100)
        self.interval = int(3.0 * self.sample_rate)
        self.two_ways_of_overlay = ['loop', 'random_position']
        self.window_size_samples = 1024

        self.decibels = decibels
        self.household_noises = household_noises
        self.pets_noises = pets_noises
        self.speech_noises = speech_noises
        self.background_music_noises = background_music_noises
        self.to_mix = to_mix

        self.to_reverb = to_reverb
        self.reverberance = reverberance
        self.hf_damping = hf_damping
        self.room_scale = room_scale
        self.stereo_depth = stereo_depth
        self.pre_delay = pre_delay
        self.wet_gain = wet_gain
        self.wet_only = wet_only
        self.reverberator = (AudioEffectsChain()
                             .reverb(reverberance=self.reverberance,
                                     hf_damping=self.hf_damping,
                                     room_scale=self.room_scale,
                                     stereo_depth=self.stereo_depth,
                                     pre_delay=self.pre_delay,
                                     wet_gain=self.wet_gain,
                                     wet_only=self.wet_only)
                             )

    # collect_chunks and get_speech_timestamps from https://github.com/snakers4/silero-vad/blob/master/utils_vad.py
    def collect_chunks(self,
                       tss: List[dict],
                       wav: torch.Tensor):
        chunks = []
        for i in tss:
            chunks.append(wav[i['start']: i['end']])
        return torch.cat(chunks)

    def get_speech_timestamps(self,
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
                              progress_tracking_callback: Callable[[float], None] = None):
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

    def tensor_normalization(self,
                             input_tensor: torch.tensor) -> torch.tensor:
        if torch.max(torch.abs(input_tensor)).item() > 1.0:
            input_tensor /= torch.max(torch.abs(input_tensor))
        return input_tensor

    def input_data_preprocessing(self,
                                 input_data,
                                 temp_filename: str,
                                 temp_sample_rate) -> torch.tensor:

        input_format = type(input_data).__name__
        final_filename = temp_filename
        defined_sample_rate = temp_sample_rate

        try:
            if input_format == 'str':
                final_filename = Path(input_data).stem
                preprocessed_data, defined_sample_rate = torchaudio.load(input_data, normalize=True)
            elif input_format == 'Tensor':
                preprocessed_data = input_data
                if preprocessed_data.size()[0] != 1:
                    preprocessed_data = torch.unsqueeze(preprocessed_data, 0)
            elif input_format == 'ndarray':
                preprocessed_data = np.float32(input_data)
                preprocessed_data = torch.from_numpy(preprocessed_data)
                if preprocessed_data.size()[0] != 1:
                    preprocessed_data = torch.unsqueeze(preprocessed_data, 0)
            else:
                preprocessed_data = (f'Expected str, tensor or numpy.ndarray format,'
                                     f'got {input_format}')
                return preprocessed_data, final_filename
        except BaseException as err:
            preprocessed_data = str(err)
            return preprocessed_data, final_filename

        preprocessed_data = self.tensor_normalization(preprocessed_data)
        preprocessed_data.to(self.device)
        self.resampler.orig_freq = defined_sample_rate
        preprocessed_data = self.resampler(preprocessed_data)

        return preprocessed_data, final_filename

    def reverberate(self,
                    audio_to_reverb_input,
                    file_original_sample_rate: int = 16000) -> dict:

        reverbed_result = {}
        generated_filename = ''.join(random.choices(string.ascii_lowercase, k=5))

        if self.to_reverb:

            audio_to_reverb, filename = self.input_data_preprocessing(input_data=audio_to_reverb_input,
                                                                      temp_filename=generated_filename,
                                                                      temp_sample_rate=file_original_sample_rate)
            if type(audio_to_reverb).__name__ == 'str':
                reverbed_result[filename] = audio_to_reverb
                return reverbed_result
            reverbed_audio_name = f'{filename}_reverbed.wav'
            try:
                reverbed_audio_array = self.reverberator(audio_to_reverb[0].numpy(),
                                                         sample_in=self.sample_rate,
                                                         sample_out=self.sample_rate)
                reverbed_audio_array = np.float32(reverbed_audio_array)
                reverbed_audio_tensor = torch.unsqueeze(torch.from_numpy(reverbed_audio_array), 0)
                reverbed_audio_tensor = self.tensor_normalization(reverbed_audio_tensor)
                reverbed_result[reverbed_audio_name] = reverbed_audio_tensor
            except BaseException as err:
                reverbed_result[reverbed_audio_name] = str(err)

        return reverbed_result

    def augmentation_overlay(self,
                             original_audio_tensor: torch.tensor,
                             prepared_noise_audio_tensor: torch.tensor) -> torch.tensor:
        overlay_result = None
        random_choice = random.choice(self.two_ways_of_overlay)
        audio_to_augment_duration = len(original_audio_tensor[0]) / float(self.sample_rate)
        noise_file_duration = len(prepared_noise_audio_tensor[0]) / float(self.sample_rate)
        audio_to_augment_tensor_length = len(original_audio_tensor[0])
        original_audio_tensor.to(self.device)

        if random_choice == 'loop':
            repeat_times = math.ceil(audio_to_augment_duration / noise_file_duration)
            loop_cat = prepared_noise_audio_tensor.repeat(1, repeat_times)
            loop_cat_to_insert = torch.unsqueeze(loop_cat[0][0:audio_to_augment_tensor_length], 0)
            loop_cat_to_insert.to(self.device)
            overlay_result = self.overlayer(original_audio_tensor,
                                            noise=loop_cat_to_insert,
                                            snr=torch.tensor([self.decibels]))

        elif random_choice == 'random_position':
            bound = 1 - (noise_file_duration / audio_to_augment_duration)
            if bound <= 0:
                random_position = 0
            else:
                random_position = random.uniform(0.01, bound)
            start_position = int(random_position * (audio_to_augment_tensor_length - 1))
            end_position = int(start_position + (len(prepared_noise_audio_tensor[0])))
            if end_position > audio_to_augment_tensor_length:
                difference = end_position - audio_to_augment_tensor_length
                cut_position = len(prepared_noise_audio_tensor[0]) - difference - 1
                prepared_noise_audio_tensor = prepared_noise_audio_tensor[0][0:cut_position]
                end_position = audio_to_augment_tensor_length - 1
            padded_noise_tensor = torch.zeros(1, audio_to_augment_tensor_length)
            padded_noise_tensor[0][start_position:end_position] = prepared_noise_audio_tensor
            padded_noise_tensor.to(self.device)
            overlay_result = self.overlayer(original_audio_tensor,
                                            noise=padded_noise_tensor,
                                            snr=torch.tensor([self.decibels]))

        return overlay_result

    def signal_energy_noise_search(self,
                                   audio_noise_numpy_array: np.ndarray) -> np.ndarray:
        input_audio_bytes = np.asarray(audio_noise_numpy_array * 32768.0, dtype=np.int16).tobytes()
        n_data = len(input_audio_bytes)
        sound_signal = np.empty((int(n_data / 2),))
        for ind in range(sound_signal.shape[0]):
            sound_signal[ind] = float(struct.unpack('<h', input_audio_bytes[(ind * 2):(ind * 2 + 2)])[0])
        frequencies_axis, time_axis, spectrogram = signal.spectrogram(sound_signal,
                                                                      fs=self.sample_rate,
                                                                      window='hamming',
                                                                      nperseg=self.nperseg,
                                                                      noverlap=0,
                                                                      scaling='spectrum',
                                                                      mode='psd')
        frame_size = int(round(0.001 * float(self.sample_rate)))
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
        minimums = [i * self.nperseg for i in energy_minimums_indices]
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
        max_maximum_index = max_maximum_index * self.nperseg

        upper_time_bound = max_maximum_index + self.interval
        lower_time_bound = max_maximum_index - self.interval
        start_minimums = [i for i in minimums if (i >= lower_time_bound) and (i < max_maximum_index)]
        finish_minimums = [i for i in minimums if (i <= upper_time_bound) and (i > max_maximum_index)]
        startpoint = min(start_minimums)
        finishpoint = max(finish_minimums)
        noise_fragment = audio_noise_numpy_array[startpoint:finishpoint]
        return noise_fragment

    def augmentate(self,
                   audio_to_augment_input,
                   file_original_sample_rate: int = 16000) -> dict:
        noises_types_dict = {"household_noises": self.household_noises,
                             "pets_noises": self.pets_noises,
                             "speech_noises": self.speech_noises,
                             "background_music_noises": self.background_music_noises}

        augmented_audiofiles = {}
        generated_filename = ''.join(random.choices(string.ascii_lowercase, k=5))

        noise_to_mix_tensors = []

        if True in list(noises_types_dict.values()):

            # audio
            (audio_to_augment_tensor,
             filename) = self.input_data_preprocessing(input_data=audio_to_augment_input,
                                                       temp_filename=generated_filename,
                                                       temp_sample_rate=file_original_sample_rate)
            if type(audio_to_augment_tensor).__name__ == 'str':
                augmented_audiofiles[filename] = audio_to_augment_tensor
                return augmented_audiofiles

            if self.to_mix:
                noise_to_mix_tensors.append(audio_to_augment_tensor)

            for noise_type in noises_types_dict.keys():
                augmented_audio_filename = f'{filename}_{noise_type}_{str(int(self.decibels))}.wav'
                try:
                    if noises_types_dict[noise_type]:

                        # noise processing
                        noises_source = self.noises_dataset[noise_type]
                        dataset_last_row_index = noises_source.num_rows - 1
                        noise_to_mix_id = random.choice(range(0, dataset_last_row_index))
                        noise_to_mix_array = noises_source[noise_to_mix_id]['audio']['array']
                        noise_file_duration = len(noise_to_mix_array) / float(self.sample_rate)

                        noise_to_mix_tensor = None

                        if noise_file_duration > 3.0:
                            if noise_type == "speech_noises":
                                noise_to_mix_tensor = torch.from_numpy(np.float32(noise_to_mix_array))
                                noise_to_mix_tensor = self.tensor_normalization(noise_to_mix_tensor)
                                noise_to_mix_tensor.to(self.device)
                                if self.model is not None:
                                    speech_timestamps = self.get_speech_timestamps(input_audio=noise_to_mix_tensor,
                                                                                   silero_vad_model=self.model,
                                                                                   sampling_rate_value=self.sample_rate)
                                    if len(speech_timestamps) >= 1:
                                        noise_to_mix_tensor = self.collect_chunks(speech_timestamps,
                                                                                  noise_to_mix_tensor)
                                noise_to_mix_tensor = torch.unsqueeze(noise_to_mix_tensor, 0)
                            elif noise_type != "speech_noises":
                                noise_to_mix_array = self.signal_energy_noise_search(noise_to_mix_array)
                                noise_to_mix_array = np.float32(noise_to_mix_array)
                                noise_to_mix_tensor = torch.unsqueeze(torch.from_numpy(noise_to_mix_array), 0)
                        elif noise_file_duration <= 3.0:
                            noise_to_mix_array = np.float32(noise_to_mix_array)
                            noise_to_mix_tensor = torch.unsqueeze(torch.from_numpy(noise_to_mix_array), 0)

                        noise_to_mix_tensor = self.tensor_normalization(noise_to_mix_tensor)

                        # augmentation
                        if self.to_mix:
                            noise_to_mix_tensors.append(noise_to_mix_tensor)

                        augmented_audio_tensor = self.augmentation_overlay(
                            original_audio_tensor=audio_to_augment_tensor,
                            prepared_noise_audio_tensor=noise_to_mix_tensor)
                        augmented_audio_tensor = self.tensor_normalization(augmented_audio_tensor)
                        augmented_audiofiles[augmented_audio_filename] = augmented_audio_tensor

                except BaseException as err:
                    augmented_audiofiles[augmented_audio_filename] = str(err)

            if self.to_mix:
                filename_mixed = f'{filename}_mixed.wav'
                try:
                    if len(noise_to_mix_tensors) < 3:
                        augmented_audiofiles[
                            filename_mixed] = 'You chose no noise types to mix or you chose only one type.'
                    elif len(noise_to_mix_tensors) >= 3:
                        noises_mix = noise_to_mix_tensors[1]
                        for n in noise_to_mix_tensors[2::]:
                            noises_mix.to(self.device)
                            noises_mix_length = len(noises_mix[0])
                            end_position = len(n[0])
                            if end_position > noises_mix_length:
                                difference = end_position - noises_mix_length
                                cut_position = end_position - difference - 1
                                n = n[0][0:cut_position]
                                end_position = noises_mix_length - 1
                            template_zero_tensor = torch.zeros(1, noises_mix_length)
                            template_zero_tensor[0][0:end_position] = n
                            template_zero_tensor.to(self.device)
                            noises_mix = self.overlayer(noises_mix,
                                                        noise=template_zero_tensor,
                                                        snr=torch.tensor([0]))

                        mixed_audio_tensor = self.augmentation_overlay(original_audio_tensor=noise_to_mix_tensors[0],
                                                                       prepared_noise_audio_tensor=noises_mix)
                        mixed_audio_tensor = self.tensor_normalization(mixed_audio_tensor)
                        augmented_audiofiles[filename_mixed] = mixed_audio_tensor
                except BaseException as err:
                    augmented_audiofiles[filename_mixed] = str(err)

        return augmented_audiofiles
