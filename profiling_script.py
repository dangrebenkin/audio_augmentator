import torch
import torchaudio
from audio_augmentator import Augmentator

augmentator_object = Augmentator(
    noises_dataset='/mnt/c/Users/LimpWinter/Documents/Projects/audio_augmentator/noises_dataset',
    silero_vad_model_path='/mnt/c/Users/LimpWinter/Documents/Projects/audio_augmentator/silero_vad.jit',
    decibels=10.0,
    household_noises=True,
    pets_noises=True,
    speech_noises=True,
    background_music_noises=True,
    to_mix=True
)
# @profile
def augmentation():


    # audio_to_augment, org_sr = torchaudio.load(audio_to_augment, normalize=True)  # | torch.tensor

    org_sr = 16_000
    for i in range(10):
        audio_to_augment = torch.randn((1, 64_000))
        augmentation_results_dict = augmentator_object.augmentate(audio_to_augment, org_sr)
        # for i in augmentation_results_dict.keys():
        #     torchaudio.save(f'temp/{i}', augmentation_results_dict[i], sample_rate=16_000, bits_per_sample=16)

augmentation()