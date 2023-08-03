import pickle
import torchaudio
from audio_augmentator.Augmentator import Augmentator

augmentator_object = Augmentator(noises_dataset='./data',
                                 to_augment=True,
                                 decibels=20.0,
                                 household_noises=True,
                                 pets_noises=True,
                                 speech_noises=True,
                                 background_music_noises=True)
augmentation_results_dict = augmentator_object.augmentate('test.wav')
for i in augmentation_results_dict.keys():
    augmented_audio_tensor = pickle.loads(augmentation_results_dict[i])
    torchaudio.save(f'{i}', augmented_audio_tensor, 16000)
