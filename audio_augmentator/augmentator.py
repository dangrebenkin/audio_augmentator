import io
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import soundfile as sf
from datasets import Dataset, DatasetDict
from pysndfx import AudioEffectsChain

from .base_noise_dataset import get_composed_dataset
from .utils import tensor_normalization


class Augmentator:

    def __init__(
            self,
            noises_dataset: str = None,
            random_seed: int = 42,
            num_workers: int = os.cpu_count(),
            sample_rate: int = 16000,
            snr_db: float = 10.0,
            device: str = "cpu"
    ):
        """Augmentator tool constructor 

        Args:
            noises_dataset (str): Path to noises dataset with format (root_dir/ noise_type/ <wav name>.wav)
            num_workers (int, optional): Processes amount for Dataset processing. Defaults to os.cpu_count().
            sample_rate (int, optional): Audio sampling rate. Defaults to 16000.
            snr_db (float, optional): Signal-to-Noise-Ratio (SNR) for augmentation (in dB). Defaults to 10.0.
            device (str, optional): Device choice for tensor processing. Defaults to "cpu".

        Raises:
            ValueError: raises if noises_dataset was not found
        """
        self.sample_rate = sample_rate
        self.num_workers = num_workers
        self.device = device
        self.randomizer = np.random.default_rng(random_seed)

        # augmentation preset
        if not Path(noises_dataset).exists():
            raise ValueError(f'Path {noises_dataset} does not exist!')
        if noises_dataset is not None:
            self.noises_dataset = get_composed_dataset(noises_dataset)
            self.noises_types_dict = {key: False for key in self.noises_dataset.keys()}
        else:
            self.noises_dataset = None
            self.noises_types_dict = None
        self.overlayer = torchaudio.transforms.AddNoise().to(self.device)
        self.loudness_estimator = torchaudio.transforms.Loudness(self.sample_rate)
        self.two_ways_of_overlay = {'loop': True, 
                                    'random_position': True}
        self.window_size_samples = 1024
        self.signal_to_noise_db = snr_db
        self.to_mix = False

        # reverberator tool
        self.reverberator = (AudioEffectsChain().reverb())
        self.set_reverberator_settings(sample_rate=16000)

    def set_reverberator_settings(self,
                                  sample_rate: int,
                                  reverberation_preset: str = 'custom',
                                  reverberance: int = 50,
                                  hf_damping: int = 50,
                                  room_scale: int = 100,
                                  stereo_depth: int = 100,
                                  pre_delay: int = 0,
                                  wet_gain: int = 0,
                                  wet_only: bool = False):
        """Reverberation (pysndfx) settings change

        Args:
            sample_rate (int): Audio sampling rate
            reverberance (int, optional): Controls the amount/intensity of reverb 
                effect [0, 100]. Defaults to 50.
            hf_damping (int, optional): High-frequency damping - controls how quickly 
                high frequencies decay [0, 100]. Defaults to 50.
            room_scale (int, optional): Simulates the size of the 
                acoustic space [0, 100]. Defaults to 100.
            stereo_depth (int, optional): Controls the stereo width of 
                the reverb effect [0, 100]. Defaults to 100.
            pre_delay (int, optional): Delay before reverb begins (in milliseconds). Defaults to 0.
            wet_gain (int, optional): Controls the volume level (in dB) of 
                the reverb effect [-10, 10]. Defaults to 0.
            wet_only (bool, optional): If True, outputs only the reverb signal. Defaults to False.
        """
        self.sample_rate = sample_rate

        if reverberation_preset == 'custom':
            self.reverberator = (
                AudioEffectsChain().reverb(
                    reverberance=reverberance,
                    hf_damping=hf_damping,
                    room_scale=room_scale,
                    stereo_depth=stereo_depth,
                    pre_delay=pre_delay,
                    wet_gain=wet_gain,
                    wet_only=wet_only)
                )
        elif reverberation_preset == 'loud':
            self.reverberator = (
                AudioEffectsChain().reverb(
                    reverberance=100,
                    hf_damping=20,
                    room_scale=100,
                    stereo_depth=100,
                    pre_delay=50,
                    wet_gain=3,
                    wet_only=wet_only)
                )
        elif reverberation_preset == 'medium':
            self.reverberator = (
                AudioEffectsChain().reverb(
                    reverberance=60,      
                    hf_damping=40,        
                    room_scale=60,  
                    stereo_depth=70,   
                    pre_delay=25,
                    wet_gain=0,   
                    wet_only=wet_only)
                )
        elif reverberation_preset == 'quiet':
            self.reverberator = (
                AudioEffectsChain().reverb(
                    reverberance=40,     
                    hf_damping=50,       
                    room_scale=40,        
                    stereo_depth=40,      
                    pre_delay=10,       
                    wet_gain=-4,      
                    wet_only=wet_only)
                )
        else:
            raise ValueError(f'Preset {reverberation_preset} is not avaliable, \
                             only presets "custom", "loud", "medium" and "quiet" are supported.')
    
    def set_augmentator_settings(self,
                                 snr_db: int,
                                 sample_rate: int,
                                 noises: Dict[str, bool],
                                 to_mix: bool = False,
                                 loop_overlay: bool = True,
                                 random_position_overlay: bool = True):
        """Setter for augmentation settings change

        Args:
            snr_db (int): SNR in dB
            sample_rate (int): Audio sampling rate
            noises (dict): Dict with items like {'noise type from noises dataset': bool}
            to_mix (bool): Get audio, which is augmented with more than 1 noise 
                (there must be set True for >= 2 type of noises in noises dict)
            loop_overlay (bool): augmentation overlay pattern that applies noise throughout 
            the complete duration of the audio sample
            random_position_overlay (bool): augmentation overlay pattern that applies noise 
            to the single segment of the audio sample
        """
        self.signal_to_noise_db = snr_db
        self.sample_rate = sample_rate
        if not loop_overlay and not random_position_overlay:
            raise ValueError('One of the overlay patterns must be selected!')
        else:
            self.two_ways_of_overlay['loop'] = loop_overlay
            self.two_ways_of_overlay['random_position'] = random_position_overlay
        self.noises_types_dict = {key: False for key in self.noises_dataset.keys()}
        for key, value in noises.items():
            if key not in self.noises_types_dict.keys():
                raise ValueError(f'The {key} type of noises is not presented in provided noises dataset!')
            if not isinstance(value, bool):
                raise ValueError(f'The value of {key} must be bool type, not {type(value)}')
        self.noises_types_dict.update(noises)
        if to_mix:
            number_of_noises_to_mix = len([k for k, v in self.noises_types_dict.items() if v])
            assert number_of_noises_to_mix >= 2, \
            f"To mix noises you must choose at least 2 of them. Got {number_of_noises_to_mix}."
            self.to_mix = to_mix

    def align_loudness_level(self, 
                             audio: torch.Tensor, 
                             noise: torch.Tensor) -> torch.Tensor:
        if self.sample_rate != 16000:
            self.loudness_estimator = torchaudio.transforms.Loudness(self.sample_rate)
        audio_loudness_db = self.loudness_estimator(audio).item()
        noise_loudness_db = self.loudness_estimator(noise).item()
        gain_db = audio_loudness_db - noise_loudness_db
        gain_linear = 10**(gain_db / 20)
        noise_aligned = noise * gain_linear
        return noise_aligned
    
    def audio_compression(self,
                          audio_sample: Union[torch.Tensor, dict]):
        if not isinstance(audio_sample, torch.Tensor):
            waveform = np.float32(audio_sample['audio']['array'])
        else:
            waveform = np.float32(torch.squeeze(audio_sample).numpy())
        sf.write(
            buffer := io.BytesIO(), 
            waveform, 
            samplerate=self.sample_rate, 
            format='mp3',
            bitrate_mode='CONSTANT', 
            compression_level=.99
        )
        buffer.seek(0)
        waveform, _ = torchaudio.load(buffer)
        if not isinstance(audio_sample, torch.Tensor):
            audio_sample['compressed'] = waveform
            return audio_sample
        else:
            return waveform

    def input_data_preprocessing(
        self,
        input_data: Union[str, np.ndarray, torch.Tensor, 
                          Dataset, DatasetDict]) -> Tuple[torch.Tensor, str]:

        input_format = type(input_data).__name__

        if input_format == 'str':
            if not Path(input_data).exists():
                raise ValueError(f'Path {input_data} does not exist!')
            preprocessed_data, _ = torchaudio.load(input_data, normalize=True)
        elif input_format == 'Tensor':
            preprocessed_data = input_data
            if preprocessed_data.size()[0] != 1:
                preprocessed_data = torch.unsqueeze(preprocessed_data, 0)
        elif input_format == 'ndarray':
            preprocessed_data = np.float32(input_data)
            preprocessed_data = torch.from_numpy(preprocessed_data)
            if preprocessed_data.size()[0] != 1:
                preprocessed_data = torch.unsqueeze(preprocessed_data, 0)
        elif input_format == 'Dataset' or 'DatasetDict':
            return input_data
        else:
            raise ValueError('Expected str or tensor or numpy.ndarray or datasets.Dataset' 
                             f'or datasets.DatasetDict format, got {input_format}')

        preprocessed_data = tensor_normalization(preprocessed_data)
        preprocessed_data.to(self.device)

        return preprocessed_data
    
    def reverberate(self,
                    audio_sample: Union[torch.Tensor, dict]) -> Union[torch.Tensor, dict]:
        if not isinstance(audio_sample, torch.Tensor):
            audio_array = np.float32(audio_sample['audio']['array'])
            reverbed_audio_array = self.reverberator(
                audio_array,
                sample_in=self.sample_rate,
                sample_out=self.sample_rate
            )
        else:
            reverbed_audio_array = self.reverberator(
                np.float32(audio_sample.numpy()),
                sample_in=self.sample_rate,
                sample_out=self.sample_rate
            )
        reverbed_audio_array = np.float32(reverbed_audio_array)
        reverbed_audio_tensor = torch.unsqueeze(torch.from_numpy(reverbed_audio_array), 0)
        if not isinstance(audio_sample, torch.Tensor):
            audio_sample['reverbed'] = reverbed_audio_tensor
            return audio_sample
        return reverbed_audio_tensor

    def augmentation_overlay(
            self,
            original_audio_tensor: torch.Tensor,
            prepared_noise_audio_tensor: torch.Tensor,
            overlay_way: str = None,
            snr: float = None
    ) -> torch.Tensor:
        
        """Augmentation overlay method

        Noise audio recordings can be "mixed" in various combinations with an adjustable signal-to-noise ratio (in decibels)
        using one of two methods, which is selected randomly (using tools from the random library):

        1. If the "loop" method is selected as a result of random choice, then the number of necessary repetitions of the 
        noise audio signal/audio fragment is calculated, which should be no less than the input audio signal in duration. 
        After that, a new noise audio signal is created by concatenating the calculated number of noise audio signals/audio fragments;

        2. If the "random position" method is selected as a result of random choice, then the time moment of the beginning 
        and end of the noise "overlay" is calculated randomly, taking into account the duration of the input audio signal;

        3. Then, in each of the methods, the noise audio signal is "overlaid" on the input audio signal, taking into account 
        its duration (the new noise audio signal is cut off at the end of the input audio signal) and the signal-to-noise 
        ratio selected by the user.

        Returns:
            torch.Tensor: augmented audio
        """

        overlay_result: Optional[torch.Tensor] = None

        if overlay_way is None:
            overlay_way_ = self.randomizer.choice([k for k, v in self.two_ways_of_overlay.items() if v])
        else:
            overlay_way_ = overlay_way
        
        audio_to_augment_duration = len(original_audio_tensor[0]) / float(self.sample_rate)
        noise_file_duration = len(prepared_noise_audio_tensor[0]) / float(self.sample_rate)
        audio_to_augment_tensor_length = len(original_audio_tensor[0])
        original_audio_tensor.to(self.device)

        if overlay_way_ == 'loop':
            repeat_times = math.ceil(audio_to_augment_duration / noise_file_duration)
            loop_cat = prepared_noise_audio_tensor.repeat(1, repeat_times)
            loop_cat_to_insert = torch.unsqueeze(loop_cat[0][0:audio_to_augment_tensor_length], 0)
            loop_cat_to_insert.to(self.device)
            overlay_result = self.overlayer(
                original_audio_tensor,
                noise=loop_cat_to_insert,
                snr=torch.tensor([self.signal_to_noise_db if snr is None else snr])
            )

        if overlay_way_ == 'random_position':
            bound = 1 - (noise_file_duration / audio_to_augment_duration)
            if bound <= 0:
                random_position = 0
            else:
                random_position = self.randomizer.uniform(0.01, bound)
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
                snr=torch.tensor([self.signal_to_noise_db if snr is None else snr]))

        return overlay_result
    
    def augmentate(self,
                   audio_sample: Union[torch.Tensor, dict]) -> dict:
        
        noise_tensors = {}
        augmented_audio = {}
        
        for noise_type in self.noises_types_dict.keys():
            if self.noises_types_dict[noise_type]:
                noises_source = self.noises_dataset[noise_type]
                noise_to_mix_id = self.randomizer.integers(low=0, high=len(noises_source), size=1).item()
                noise_to_mix_tensor = noises_source[noise_to_mix_id]
                noise_to_mix_tensor = tensor_normalization(noise_to_mix_tensor)
                noise_tensors[noise_type] = noise_to_mix_tensor

        if self.to_mix:
            """ 
            Overlay noises to the longest one and normalize every noise according to the first noise loudness.
            """
            noises_tensors = [v for k, v in noise_tensors.items()]
            noises_tensors = sorted(noises_tensors, key=lambda x: x.size(1), reverse=True)
            noises_mix = noises_tensors[0]
            noises_mix.to(self.device)
            for n in noises_tensors[1::]:
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
                template_zero_tensor = self.align_loudness_level(noises_mix, 
                                                                 template_zero_tensor)
                noises_mix = self.augmentation_overlay(noises_mix, template_zero_tensor, 'loop', 0)
            noise_tensors['mixed'] = noises_mix
        
        if isinstance(audio_sample, dict):
            audio_array = np.float32(audio_sample['audio']['array'])
            audio_tensor = torch.unsqueeze(torch.from_numpy(audio_array), 0)
            audio_tensor = tensor_normalization(audio_tensor)
        else:
            audio_tensor = audio_sample
        for noise, noise_tensor in noise_tensors.items():
            noise_tensor = self.align_loudness_level(audio_tensor, noise_tensor)
            augmentation_result = self.augmentation_overlay(audio_tensor, noise_tensor)
            if augmentation_result.size()[0] != 1:
                augmentation_result = torch.unsqueeze(augmentation_result, 0)
            augmented_audio[noise] = augmentation_result
        if isinstance(audio_sample, dict):
            audio_sample.update(augmented_audio)
            return audio_sample
        else:
            return augmented_audio

    def run(
            self,            
            audio_input: Union[str, np.ndarray, torch.Tensor, Dataset, DatasetDict],
            augment: bool = False,
            reverberate: bool = False,
            compress: bool = False
    ) -> Union[Dataset, DatasetDict, dict]:
        
        processed_audio = self.input_data_preprocessing(audio_input)
        if isinstance(processed_audio, Dataset) or isinstance(processed_audio, DatasetDict):
            if self.num_workers > processed_audio.num_rows:
                self.num_workers = processed_audio.num_rows

            if augment:
                processed_audio = processed_audio.map(self.augmentate, 
                                                      batched=False, 
                                                      num_proc=self.num_workers)
            if reverberate:
                processed_audio = processed_audio.map(self.reverberate, 
                                                      batched=False, 
                                                      num_proc=self.num_workers)
            if compress:
                processed_audio = processed_audio.map(self.audio_compression, 
                                                      batched=False, 
                                                      num_proc=self.num_workers)
            return processed_audio.with_format("torch")
        else:
            processed_audio_dict: dict = {}
            if augment:
                processed_audio_dict.update(self.augmentate(processed_audio))
            if reverberate:
                processed_audio_dict['reverbed'] = self.reverberate(processed_audio)
            if compress:
                processed_audio_dict['compressed'] = self.audio_compression(processed_audio)
            return processed_audio_dict

