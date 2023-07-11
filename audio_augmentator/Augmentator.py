import pydub
import random
import struct
import librosa
import numpy as np
from pathlib import Path
from scipy import signal
from base64 import b64encode
from datasets import DatasetDict
from pysndfx import AudioEffectsChain


class Augmentator:

    def __init__(self,

                 noises_dataset: str,

                 to_augment: bool = False,
                 to_mix: bool = False,
                 decibels: float = 10.0,
                 household_noises: bool = False,
                 pets_noises: bool = False,
                 speech_noises: bool = False,
                 background_music_noises: bool = False,

                 to_reverb: bool = False,
                 reverberance: int = 50,
                 hf_damping: int = 50,
                 room_scale: int = 100,
                 stereo_depth: int = 100,
                 pre_delay: int = 20,
                 wet_gain: int = 0,
                 wet_only: bool = False
                 ):

        self.noises_dataset = DatasetDict.load_from_disk(noises_dataset)
        self.sample_rate = 16000
        self.nperseg = int(self.sample_rate / 100)
        self.interval = int(3.0 * self.sample_rate)
        self.two_ways_of_overlay = ['loop', 'random_position']

        self.to_augment = to_augment
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

    def reverberate(self,
                    audio_to_reverb_path: str,
                    b64encode_output: bool = False) -> dict:
        reverberator = (
            AudioEffectsChain()
            .reverb(reverberance=self.reverberance,
                    hf_damping=self.hf_damping,
                    room_scale=self.room_scale,
                    stereo_depth=self.stereo_depth,
                    pre_delay=self.pre_delay,
                    wet_gain=self.wet_gain,
                    wet_only=self.wet_only)
        )

        reverbed_result = {}
        filename = Path(audio_to_reverb_path).stem

        if self.to_reverb:
            reverbed_audio_name = f'{filename}_reverbed.wav'
            try:
                audio_to_reverb, _ = librosa.load(audio_to_reverb_path, sr=self.sample_rate)
                reverbed_audio_array = reverberator(audio_to_reverb,
                                                    sample_in=self.sample_rate,
                                                    sample_out=self.sample_rate)
                reverbed_audio_array = np.round(reverbed_audio_array * 32767.0).astype(np.int16)
                reverbed_audio_object = pydub.AudioSegment(data=reverbed_audio_array.tobytes(),
                                                           sample_width=2,
                                                           frame_rate=self.sample_rate,
                                                           channels=1)
                reverbed_audio_bytes = reverbed_audio_object.get_array_of_samples().tobytes()
                if b64encode_output:
                    reverbed_audio_bytes = b64encode(reverbed_audio_bytes).decode("utf-8")
                reverbed_result[reverbed_audio_name] = reverbed_audio_bytes
            except BaseException as err:
                reverbed_result[reverbed_audio_name] = str(err)
        return reverbed_result

    def augmentation_overlay(self,
                             original_audio_object: pydub.audio_segment.AudioSegment,
                             prepared_noise_audio_object: pydub.audio_segment.AudioSegment) -> pydub.audio_segment.AudioSegment:
        overlay_result = None
        random_choice = random.choice(self.two_ways_of_overlay)
        original_audio_duration = original_audio_object.duration_seconds
        noise_audio_duration = prepared_noise_audio_object.duration_seconds
        if random_choice == 'loop':
            overlay_result = original_audio_object.overlay(prepared_noise_audio_object,
                                                           loop=True)
        elif random_choice == 'random_position':
            bound = 1 - (noise_audio_duration / original_audio_duration)
            random_position = random.uniform(0.01, bound)
            overlay_result = original_audio_object.overlay(prepared_noise_audio_object,
                                                           position=random_position * len(original_audio_object))
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
        energy_minimums = []
        for i in range(len(energy_values) - 1):
            if (energy_values[i] < energy_values[i - 1]) and (energy_values[i] < energy_values[i + 1]):
                energy_minimums.append(i)
        energy_minimums.append(len(energy_values) - 1)
        minimums = [i * self.nperseg for i in energy_minimums]
        if minimums[0] != 0:
            minimums.insert(0, 0)

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
                   audio_to_augment_path: str,
                   b64encode_output: bool = False) -> dict:
        noises_types_dict = {"household_noises": self.household_noises,
                             "pets_noises": self.pets_noises,
                             "speech_noises": self.speech_noises,
                             "background_music_noises": self.background_music_noises}

        augmented_audiofiles = {}
        filename = Path(audio_to_augment_path).stem
        noise_to_mix_objects = []

        if self.to_augment:

            # audio
            audio_to_augment, _ = librosa.load(audio_to_augment_path, sr=self.sample_rate)
            good_audio_array = np.round(audio_to_augment * 32767.0).astype(np.int16)
            audio_to_augment_object = pydub.AudioSegment(data=good_audio_array.tobytes(),
                                                         sample_width=2,
                                                         frame_rate=self.sample_rate,
                                                         channels=1)
            audio_to_augment_object.set_frame_rate(self.sample_rate)
            if self.to_mix:
                noise_to_mix_objects.append(audio_to_augment_object)

            for noise_type in noises_types_dict.keys():
                augmented_audio_filename = f'{filename}_{noise_type}_{str(int(self.decibels))}.wav'
                try:
                    if noises_types_dict[noise_type]:

                        # noise
                        noises_source = self.noises_dataset[noise_type]
                        dataset_size = noises_source.num_rows
                        noise_to_mix_id = random.choice(range(0, dataset_size))
                        noise_to_mix_original_array = noises_source['audio'][noise_to_mix_id]['array']
                        noise_file_duration = len(noise_to_mix_original_array) / float(self.sample_rate)
                        if noise_file_duration > 3.0:
                            noise_to_mix_original_array = self.signal_energy_noise_search(noise_to_mix_original_array)
                        noise_to_mix_array = np.round(noise_to_mix_original_array * 32768.0).astype(np.int16)
                        noise_to_mix_object = pydub.AudioSegment(data=noise_to_mix_array.tobytes(),
                                                                 sample_width=2,
                                                                 frame_rate=self.sample_rate,
                                                                 channels=1)
                        amplitude_difference = audio_to_augment_object.dBFS - noise_to_mix_object.dBFS
                        noise_to_mix_object = noise_to_mix_object.apply_gain(amplitude_difference)
                        noise_to_mix_object = noise_to_mix_object.apply_gain(-self.decibels)

                        # augmentation
                        if self.to_mix:
                            noise_to_mix_objects.append(noise_to_mix_object)
                        elif not self.to_mix:
                            augmented_audio_object = self.augmentation_overlay(
                                original_audio_object=audio_to_augment_object,
                                prepared_noise_audio_object=noise_to_mix_object)
                            augmented_audio_bytes = augmented_audio_object.get_array_of_samples().tobytes()
                            if b64encode_output:
                                augmented_audio_bytes = b64encode(augmented_audio_bytes).decode("utf-8")
                            augmented_audiofiles[augmented_audio_filename] = augmented_audio_bytes
                except BaseException as err:
                    augmented_audiofiles[augmented_audio_filename] = str(err)

            if self.to_mix:
                try:
                    if len(noise_to_mix_objects) < 3:
                        augmented_audiofiles['mixed'] = 'You chose no noise types to mix or you chose only one type.'
                    elif len(noise_to_mix_objects) >= 3:
                        noises_mix = noise_to_mix_objects[1]
                        for n in noise_to_mix_objects[2::]:
                            noises_mix = noises_mix.overlay(n, loop=True)
                        mixed_audio_object = self.augmentation_overlay(original_audio_object=noise_to_mix_objects[0],
                                                                       prepared_noise_audio_object=noises_mix)
                        augmented_audio_bytes = mixed_audio_object.get_array_of_samples().tobytes()
                        if b64encode_output:
                            augmented_audio_bytes = b64encode(augmented_audio_bytes).decode("utf-8")
                        augmented_audiofiles['mixed'] = augmented_audio_bytes
                except BaseException as err:
                    augmented_audiofiles['mixed'] = str(err)
        return augmented_audiofiles
