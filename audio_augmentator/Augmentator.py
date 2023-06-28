from base64 import b64encode

import librosa
import numpy as np
import pydub
import random
from datasets import DatasetDict
from pysndfx import AudioEffectsChain


class Augmentator:

    def __init__(self,
                 noises_dataset: str = None,
                 to_augment: bool = False,
                 decibels: float = 10.0,
                 household_noises: bool = False,
                 pets_noises: bool = False,
                 speech_noises: bool = False,
                 background_music_noise: bool = False,

                 to_reverb: bool = False,
                 reverberance: int = 50,
                 hf_damping: int = 50,
                 room_scale: int = 100,
                 stereo_depth: int = 100,
                 pre_delay: int = 20,
                 wet_gain: int = 0,
                 wet_only: bool = False):

        self.to_augment = to_augment
        self.to_reverb = to_reverb
        self.noise_decibels = decibels
        self.household_noises = household_noises
        self.pets_noises = pets_noises
        self.speech_noises = speech_noises
        self.background_music_noise = background_music_noise

        if self.to_reverb:
            self.reverberator = (
                AudioEffectsChain()
                .reverb(reverberance=reverberance,
                        hf_damping=hf_damping,
                        room_scale=room_scale,
                        stereo_depth=stereo_depth,
                        pre_delay=pre_delay,
                        wet_gain=wet_gain,
                        wet_only=wet_only))

        if self.to_augment:
            self.noises_dataset = DatasetDict.load_from_disk(noises_dataset)
            self.noise_types = list(self.noises_dataset.keys())

    def reverberate(self,
                    audio_array: np.ndarray,
                    b64encode_output: bool = False) -> list:

        reverbed_result = []
        if self.to_reverb:
            try:
                good_audio_array = np.round(audio_array * 32767.0).astype(np.int16)
                reverbed_audio_array = self.reverberator(good_audio_array,
                                                         channels_out=1)
                reverbed_audio_bytes = reverbed_audio_array.tobytes()
                if b64encode_output:
                    reverbed_audio_bytes = b64encode(reverbed_audio_bytes).decode("utf-8")

                reverbed_result.append(reverbed_audio_bytes)
            except BaseException as err:
                reverbed_result.append(str(err))
        return reverbed_result

    def augmentate(self,
                   audio_path: str,
                   b64encode_output: bool = False) -> list:

        augmented_audiofiles = []
        parameters = {"household_noises": self.household_noises,
                      "pets_noises": self.pets_noises,
                      "speech_noises": self.speech_noises,
                      "background_music_noises": self.background_music_noise}

        if self.to_augment:
            for noise_type in parameters.keys():
                try:
                    if parameters[noise_type]:
                        audio_to_augment, _ = librosa.load(audio_path, sr=16000)
                        good_audio_array = np.round(audio_to_augment * 32767.0).astype(np.int16)
                        audio_to_augment_object = pydub.AudioSegment(data=good_audio_array.tobytes(),
                                                                     sample_width=2,
                                                                     frame_rate=16000,
                                                                     channels=1)
                        noises_source = self.noises_dataset[noise_type]
                        dataset_size = noises_source.num_rows
                        noise_to_mix_id = random.choice(range(0, dataset_size))
                        noise_to_mix = noises_source['audio'][noise_to_mix_id]['array']
                        good_noise_to_mix_array = np.round(noise_to_mix * 32767.0).astype(np.int16)
                        noise_to_mix_object = pydub.AudioSegment(data=good_noise_to_mix_array.tobytes(),
                                                                 sample_width=2,
                                                                 frame_rate=16000,
                                                                 channels=1)
                        amplitude_difference = audio_to_augment_object.dBFS - noise_to_mix_object.dBFS
                        noise_to_mix_object = noise_to_mix_object.apply_gain(amplitude_difference)
                        noise_to_mix_object = noise_to_mix_object.apply_gain(-self.noise_decibels)
                        augmented_audio_object = audio_to_augment_object.overlay(
                            noise_to_mix_object,
                            loop=True)
                        augmented_audio_bytes = augmented_audio_object.get_array_of_samples().tobytes()
                        if b64encode_output:
                            augmented_audio_bytes = b64encode(augmented_audio_bytes).decode("utf-8")
                        augmented_audiofiles.append(augmented_audio_bytes)
                except BaseException as err:
                    augmented_audiofiles.append(str(err))
        return augmented_audiofiles
