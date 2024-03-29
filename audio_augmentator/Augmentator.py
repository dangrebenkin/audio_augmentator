import math
import os

import torch
import random
import string
import torchaudio
import numpy as np
import numpy.typing as npt
from pathlib import Path
from pysndfx import AudioEffectsChain
from typing import Dict, Union, Tuple, Optional
from .utils import (
    tensor_normalization,
    get_composed_dataset
)


class Augmentator:

    def __init__(
            self,
            noises_dataset: str,
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
            wet_only: bool = False,
            device: str = "cpu"
    ):

        assert os.path.isdir(noises_dataset), f'"{noises_dataset}" does not exist!'
        self.noises_dataset = get_composed_dataset(noises_dataset)
        self.device = device
        self.overlayer = torchaudio.transforms.AddNoise().to(self.device)
        self.sample_rate = 16000
        self.resampler = torchaudio.transforms.Resample(new_freq=self.sample_rate).to(self.device)

        self.two_ways_of_overlay = ['loop', 'random_position']
        self.window_size_samples = 1024
        self.decibels = decibels

        self.household_noises = household_noises
        self.pets_noises = pets_noises
        self.speech_noises = speech_noises
        self.background_music_noises = background_music_noises
        self.to_mix = to_mix

        if self.to_mix:
            number_of_noises_to_mix = sum([background_music_noises, household_noises, pets_noises, speech_noises])
            assert number_of_noises_to_mix >= 2, \
                f"To mix noises you must choose at least 2 of them. Got {number_of_noises_to_mix}."

        self.to_reverb = to_reverb
        self.reverberance = reverberance
        self.hf_damping = hf_damping
        self.room_scale = room_scale
        self.stereo_depth = stereo_depth
        self.pre_delay = pre_delay
        self.wet_gain = wet_gain
        self.wet_only = wet_only
        self.reverberator = (
            AudioEffectsChain().reverb(
                reverberance=self.reverberance,
                hf_damping=self.hf_damping,
                room_scale=self.room_scale,
                stereo_depth=self.stereo_depth,
                pre_delay=self.pre_delay,
                wet_gain=self.wet_gain,
                wet_only=self.wet_only
            )
        )

    def input_data_preprocessing(
            self,
            input_data: Union[str, np.ndarray, torch.Tensor],
            temp_filename: str,
            temp_sample_rate: int
    ) -> Tuple[torch.Tensor, str]:

        input_format = type(input_data).__name__
        final_filename = temp_filename
        defined_sample_rate = temp_sample_rate

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
            raise ValueError(f'Expected str, tensor or numpy.ndarray format, got {input_format}')

        preprocessed_data = tensor_normalization(preprocessed_data)
        preprocessed_data.to(self.device)
        self.resampler.orig_freq = defined_sample_rate
        preprocessed_data = self.resampler(preprocessed_data)

        return preprocessed_data, final_filename

    def reverberate(
            self,
            audio_to_reverb_input: Union[str, npt.NDArray, torch.Tensor],
            file_original_sample_rate: int = 16000
    ) -> Dict[str, torch.Tensor]:

        reverbed_result: Dict[str, torch.Tensor] = {}
        generated_filename = ''.join(random.choices(string.ascii_lowercase, k=5))

        if self.to_reverb:

            audio_to_reverb, filename = self.input_data_preprocessing(
                input_data=audio_to_reverb_input,
                temp_filename=generated_filename,
                temp_sample_rate=file_original_sample_rate
            )
            if type(audio_to_reverb).__name__ == 'str':
                reverbed_result[filename] = audio_to_reverb
                return reverbed_result
            reverbed_audio_name = f'{filename}_reverbed.wav'

            reverbed_audio_array = self.reverberator(
                audio_to_reverb[0].numpy(),
                sample_in=self.sample_rate,
                sample_out=self.sample_rate
            )
            reverbed_audio_array = np.float32(reverbed_audio_array)
            reverbed_audio_tensor = torch.unsqueeze(torch.from_numpy(reverbed_audio_array), 0)
            reverbed_audio_tensor = tensor_normalization(reverbed_audio_tensor)
            reverbed_result[reverbed_audio_name] = reverbed_audio_tensor

        return reverbed_result

    def augmentation_overlay(
            self,
            original_audio_tensor: torch.Tensor,
            prepared_noise_audio_tensor: torch.Tensor
    ) -> torch.Tensor:

        overlay_result: Optional[torch.Tensor] = None
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
            overlay_result = self.overlayer(
                original_audio_tensor,
                noise=loop_cat_to_insert,
                snr=torch.tensor([self.decibels])
            )

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
            overlay_result = self.overlayer(
                original_audio_tensor,
                noise=padded_noise_tensor,
                snr=torch.tensor([self.decibels]))

        return overlay_result

    def augmentate(
            self,
            audio_to_augment_input: Union[str, np.ndarray, torch.Tensor],
            file_original_sample_rate: int = 16000
    ) -> Dict[str, torch.Tensor]:
        """
        :param audio_to_augment_input: array or path to audio to augment
        :param file_original_sample_rate:
        :return: dictionary with augmented audio files as values and their str names as keys
        """
        noises_types_dict = {
            "household": self.household_noises,
            "pets": self.pets_noises,
            "speech": self.speech_noises,
            "background_music": self.background_music_noises
        }

        augmented_audiofiles = {}
        generated_filename = ''.join(random.choices(string.ascii_lowercase, k=5))

        noise_to_mix_tensors = []

        if True in list(noises_types_dict.values()):

            # audio
            audio_to_augment_tensor, filename = self.input_data_preprocessing(
                input_data=audio_to_augment_input,
                temp_filename=generated_filename,
                temp_sample_rate=file_original_sample_rate
            )
            if type(audio_to_augment_tensor).__name__ == 'str':
                augmented_audiofiles[filename] = audio_to_augment_tensor
                return augmented_audiofiles

            if self.to_mix:
                noise_to_mix_tensors.append(audio_to_augment_tensor)

            for noise_type in noises_types_dict.keys():
                augmented_audio_filename = f'{filename}_{noise_type}_{str(int(self.decibels))}.wav'
                try:
                    if noises_types_dict[noise_type]:

                        noises_source = self.noises_dataset[noise_type]
                        dataset_last_row_index = len(noises_source) - 1
                        noise_to_mix_id = random.randint(0, dataset_last_row_index)
                        noise_to_mix_tensor = noises_source[noise_to_mix_id]
                        noise_to_mix_tensor = tensor_normalization(noise_to_mix_tensor)

                        # augmentation
                        if self.to_mix:
                            noise_to_mix_tensors.append(noise_to_mix_tensor)

                        augmented_audio_tensor = self.augmentation_overlay(
                            original_audio_tensor=audio_to_augment_tensor,
                            prepared_noise_audio_tensor=noise_to_mix_tensor
                        )
                        augmented_audio_tensor = tensor_normalization(augmented_audio_tensor)
                        augmented_audiofiles[augmented_audio_filename] = augmented_audio_tensor

                except BaseException as err:
                    augmented_audiofiles[augmented_audio_filename] = str(err)

            if self.to_mix:
                filename_mixed = f'{filename}_mixed.wav'
                try:
                    if len(noise_to_mix_tensors) < 3:
                        assert False, "Not reachable"
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
                            noises_mix = self.overlayer(
                                noises_mix,
                                noise=template_zero_tensor,
                                snr=torch.tensor([0])
                            )

                        mixed_audio_tensor = self.augmentation_overlay(
                            original_audio_tensor=noise_to_mix_tensors[0],
                            prepared_noise_audio_tensor=noises_mix
                        )
                        mixed_audio_tensor = tensor_normalization(mixed_audio_tensor)
                        augmented_audiofiles[filename_mixed] = mixed_audio_tensor
                except BaseException as err:
                    augmented_audiofiles[filename_mixed] = str(err)

        return augmented_audiofiles
