import torchaudio

from audio_augmentator.Augmentator import Augmentator

augmentator_object = Augmentator(noises_dataset='/home/daniil/Документы/datasets/noises_corpora',
                                 silero_vad_model_path='silero_vad.jit',
                                 decibels=20.0,
                                 household_noises=True,
                                 pets_noises=True,
                                 speech_noises=True,
                                 background_music_noises=True,
                                 # to_mix=True
                                 )

# на вход можно подать звуковой файл в 3 вариантах: строковый путь
# к файлу; в формате torch.tensor; в формате numpy.ndarray
# частота дискретизации входного файла по умолчанию 16 кГц, но можно указать ее вручную,
# передав ее значение в качестве второго аргумента augmentator_object.augmentate
# (или для augmentator_object.reverberate)

audio_to_augment = 'test.wav'
audio_to_augment, org_sr = torchaudio.load(audio_to_augment, normalize=True)  # | torch.tensor
# audio_to_augment = audio_to_augment.detach().cpu().numpy() | numpy.ndarray

augmentation_results_dict = augmentator_object.augmentate(audio_to_augment, org_sr)
for i in augmentation_results_dict.keys():
    torchaudio.save(f'{i}', augmentation_results_dict[i], 16000)
