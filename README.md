# Audio_augmentator

Audio augmentation and reverberation tool.

### Usage steps

##### **1. Create _Augmentator_ object:**

```
from audio_augmentator.Augmentator import Augmentator

augmentator_object = Augmentator()
```

Set `augmentator_object` parameters depending on audio augmentation type: augmentation with different types of 
background noises (step 1.1) or reverberation effect (step 1.2).

##### **1.1 Augmentation (`Augmentator(to_augment=True)`)**

Augmentation parameters set:

* `to_augment`: enable augmentation mode (**always required for augmentation**), default = False;
* `noises_dataset`: path to noises corpora, you should [download](https://disk.yandex.ru/d/o4dEtHOtR6BGgw) it ,
  extract zip-file and write a path to extracted dataset as argument value, **always required for augmentation**;
* `decibels`: a difference (dBS) of between input audio signal volume and noise volume (e.g. if `decibels=5.0` it means
  that noise level will be 10 dBS lower than original audio volume in augmented audio), (**optional**) default = 10.0;
* `household_noises`: set True to get audio augmented with household noises (**optional**), default = False;
* `pets_noises`: set True to get audio augmented with pets noises (**optional**), default = False;
* `speech_noises`: set True to get audio augmented with speech (**optional**), default = False;
* `background_music_noise`: set True to get audio augmented with music noises (**optional**), default = False.

Example of `augmentator_object` specified for augmentation:
```
augmentator_object_1 = Augmentator(to_augment=True,
                                 noises_dataset='path/to/dataset/',
                                 decibels=5.0,
                                 household_noises=True,
                                 background_music_noise=True) 
```
The parameters set of this example will let you get two files as output: original file augmented with household noises 
and music noises. The noise from corpora is chosen randomly.

##### **1.2 Reverberation (`Augmentator(to_reverb=True)`)**

Reverberation parameters set:

* `to_reverb`: enable reverberation mode (**always required for reverberation**), default = False;
* `reverberance`: (**optional**) default = 50;
* `hf_damping`: (**optional**) default = 50;
* `room_scale`: (**optional**) default = 100;
* `stereo_depth`: (**optional**) default = 100;
* `pre_delay`: (**optional**) default = 20;
* `wet_gain`: (**optional**) default = 0;
* `wet_only`: (**optional**) default = False).

Reverberation was implemented with using pysndfx library (https://github.com/carlthome/python-audio-effects).

Example (using default reverb values):

```
augmentator_object_2 = Augmentator(to_reverb=True) 
```

##### **2. Augmentation | reverberation outputs**

Use reverberate() or augmentate() to get outputs:

```
augmented_sound = augmentator_object_1.augmentate(audio_to_augment_path='wav_to_augment.wav')
reverbed_sound = augmentator_object_2.reverberate(audio_to_reverb_path='wav_to_reverb.wav')
augmented_sound  # {'<wav_to_augment>_household_noises_5.wav': <bytes array>, '<wav_to_augment>_background_music_noise_5.wav': <bytes array>}
reverbed_sound  # {'<wav_to_reverb>_reverbed.wav': <bytes array>}
```

You can also get augmented/reverbed files as strings:

```
augmented_sound = augmentator_object_1.augmentate(audio_to_augment_path='wav_to_augment.wav',
                                                  b64encode_output=True)
reverbed_sound = augmentator_object_2.reverberate(audio_to_reverb_path='wav_to_reverb.wav',
                                                  b64encode_output=True)
augmented_sound  # {'<wav_to_augment>_household_noises_5.wav': <string>, '<wav_to_augment>_background_music_noise_5.wav': <string>}
reverbed_sound  # {'<wav_to_reverb>_reverbed.wav': <string>}
```
