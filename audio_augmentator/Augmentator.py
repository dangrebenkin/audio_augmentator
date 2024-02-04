import math
import os

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
from .utils import (
    signal_energy_noise_search,
    get_speech_timestamps,
    preprocess_other,
    preprocess_speech
)


# from line_profiler_pycharm import profile

class Augmentator:

    def __init__(
            self,
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
        assert os.path.isdir(noises_dataset), f'"{noises_dataset}" does not exist!'
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

    def augmentate(
            self,
            audio_to_augment_input,
            file_original_sample_rate: int = 16000
    ) -> dict:

        noises_types_dict = {"household": self.household_noises,
                             "pets": self.pets_noises,
                             "speech": self.speech_noises,
                             "background": self.background_music_noises}

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
                        # noise_file_duration = len(noise_to_mix_array) / float(self.sample_rate)
                        # noise_to_mix_tensor = torch.from_numpy(np.float32(noise_to_mix_array))


                        # noise_to_mix_tensor = None

                        # if noise_file_duration > 3.0:
                        #     if noise_type == "speech_noises":
                        #         noise_to_mix_tensor = torch.from_numpy(np.float32(noise_to_mix_array))
                        #         noise_to_mix_tensor = self.tensor_normalization(noise_to_mix_tensor)
                        #         noise_to_mix_tensor.to(self.device)
                        #         if self.model is not None:
                        #             speech_timestamps = get_speech_timestamps(
                        #                 input_audio=noise_to_mix_tensor,
                        #                 silero_vad_model=self.model,
                        #                 sampling_rate_value=self.sample_rate
                        #             )
                        #             if len(speech_timestamps) >= 1:
                        #                 noise_to_mix_tensor = self.collect_chunks(speech_timestamps,
                        #                                                           noise_to_mix_tensor)
                        #         noise_to_mix_tensor = torch.unsqueeze(noise_to_mix_tensor, 0)
                        #         noise_to_mix_tensor = preprocess_speech(noise_to_mix_array, self.model)
                        #     elif noise_type != "speech_noises":
                        #         noise_to_mix_array = signal_energy_noise_search(noise_to_mix_array)
                        #         noise_to_mix_array = np.float32(noise_to_mix_array)
                        #         noise_to_mix_tensor = torch.unsqueeze(torch.from_numpy(noise_to_mix_array), 0)
                        # elif noise_file_duration <= 3.0:
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
