from base64 import b64encode
from pathlib import Path
import librosa
import numpy as np
import pydub
import random
from datasets import DatasetDict
from pysndfx import AudioEffectsChain


class Augmentator:

    def __init__(self,
                 noises_dataset: str,
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
                 wet_only: bool = False,

                 sample_rate: int = 16000):

        self.sample_rate = sample_rate
        self.to_augment = to_augment
        self.to_reverb = to_reverb
        self.noise_decibels = decibels
        self.parameters = {"household_noises": household_noises,
                           "pets_noises": pets_noises,
                           "speech_noises": speech_noises,
                           "background_music_noises": background_music_noise}

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
                    audio_to_reverb_path: str,
                    b64encode_output: bool = False) -> dict:

        reverbed_result = {}
        filename = Path(audio_to_reverb_path).stem

        if self.to_reverb:
            reverbed_audio_name = f'{filename}_reverbed.wav'
            try:
                audio_to_reverb, _ = librosa.load(audio_to_reverb_path, sr=self.sample_rate)
                good_audio_array = np.round(audio_to_reverb * 32767.0).astype(np.int16)
                reverbed_audio_array = self.reverberator(good_audio_array,
                                                         channels_out=1)
                reverbed_audio_bytes = reverbed_audio_array.tobytes()
                if b64encode_output:
                    reverbed_audio_bytes = b64encode(reverbed_audio_bytes).decode("utf-8")
                reverbed_result[reverbed_audio_name] = reverbed_audio_bytes
            except BaseException as err:
                reverbed_result[reverbed_audio_name] = str(err)
        return reverbed_result

    def augmentate(self,
                   audio_to_augment_path: str,
                   b64encode_output: bool = False) -> dict:

        augmented_audiofiles = {}
        filename = Path(audio_to_augment_path).stem

        if self.to_augment:
            for noise_type in self.parameters.keys():
                augmented_audio_filename = f'{filename}_{noise_type}_{str(int(self.noise_decibels))}.wav'
                try:
                    if self.parameters[noise_type]:

                        audio_to_augment, _ = librosa.load(audio_to_augment_path, sr=self.sample_rate)
                        good_audio_array = np.round(audio_to_augment * 32767.0).astype(np.int16)
                        audio_to_augment_object = pydub.AudioSegment(data=good_audio_array.tobytes(),
                                                                     sample_width=2,
                                                                     frame_rate=16000,
                                                                     channels=1)
                        audio_to_augment_object.set_frame_rate(self.sample_rate)
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
                        augmented_audio_object = audio_to_augment_object.overlay(
                            noise_to_mix_object.apply_gain(-self.noise_decibels),
                            loop=True)
                        augmented_audio_bytes = augmented_audio_object.get_array_of_samples().tobytes()
                        if b64encode_output:
                            augmented_audio_bytes = b64encode(augmented_audio_bytes).decode("utf-8")
                        augmented_audiofiles[augmented_audio_filename] = augmented_audio_bytes
                except BaseException as err:
                    augmented_audiofiles[augmented_audio_filename] = str(err)
        return augmented_audiofiles
