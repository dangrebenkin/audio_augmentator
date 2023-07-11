
from multiprocessing import Pool, cpu_count
from base64 import b64decode
from pydub.playback import play
from functools import partial
import pydub

from audio_augmentator.Augmentator import Augmentator

cores_number = cpu_count()
augmentation_tool = Augmentator(noises_dataset='/home/user/documents/projects/audio_augmentator/noises_dataset',
                                decibels=5.0,
                                to_augment=True,
                                speech_noises=True
                                )
pool = Pool(processes=cores_number)
file_names = ['1.wav', '2.wav']
all_results = {}
for result in pool.map(partial(augmentation_tool.augmentate, b64encode_output=True), file_names):
    all_results.update(result)
pool.close()
pool.join()

for i in all_results.keys():
    print(i)
    b64_string = all_results[i]
    bytes_string = b64_string.encode('utf-8')
    bytes_string = b64decode(bytes_string)
    audio_object = pydub.AudioSegment(data=bytes_string,
                                      sample_width=2,
                                      frame_rate=16000,
                                      channels=1)
    play(audio_object)
