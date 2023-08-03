import pickle
import torchaudio
from multiprocessing import Pool, cpu_count
from audio_augmentator.Augmentator import Augmentator

cores_number = cpu_count()
augmentation_tool = Augmentator(noises_dataset='/home/user/documents/projects/audio_augmentator/noises_dataset',
                                decibels=5.0,
                                speech_noises=True
                                )
pool = Pool(processes=cores_number)
file_names = ['1.wav', '2.wav']
all_results = {}
for result in pool.map(augmentation_tool.augmentate, file_names):
    all_results.update(result)
pool.close()
pool.join()

for i in all_results.keys():
    augmented_audio_tensor = pickle.loads(all_results[i])
    torchaudio.save(f'{i}', augmented_audio_tensor, 16000)
