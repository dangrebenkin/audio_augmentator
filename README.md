# Audio Augmentator

Audio augmentation and reverberation tool.

### Installation

```
sudo apt-get install sox
git clone https://github.com/dangrebenkin/audio_augmentator
cd audio_augmentator
python setup.py install
```
**The noises dataset and silero-vad model will be downloaded automatically during installation. 
Use absolute paths of downloaded files as `noises_dataset=` and `silero_vad_model_path=` values.**

### Usage steps

##### **1. Create _Augmentator_ object:**

```
from audio_augmentator.Augmentator import Augmentator

augmentator_object = Augmentator(noises_dataset=<path to dataset>)
```

Set `augmentator_object` parameters depending on audio augmentation type: augmentation with different types of 
background noises (step 1.1) or reverberation effect (step 1.2).

##### **1.1 Augmentation (`Augmentator(to_augment=True)`)**

Augmentation parameters set:

* `decibels`: a difference (dBS) of between input audio signal volume and noise volume (e.g. if `decibels=5.0` it means
  that noise level will be 10 dBS lower than original audio volume in augmented audio), (**optional**) default = 10.0;
* `household_noises`: set True to get audio augmented with household noises (**optional**), default = False;
* `pets_noises`: set True to get audio augmented with pets noises (**optional**), default = False;
* `speech_noises`: set True to get audio augmented with speech (**optional**), default = False;
* `background_music_noises`: set True to get audio augmented with music noises (**optional**), default = False.
* `to_mix`: set True to get audio mixed with several types of noises (you should set True to at least two types 
of noises to get the result)(**optional**), default = False.

**(!) You have to set True to one of the noises types (`household_noises`, `pets_noises`, `speech_noises`, `background_music_noises`) parameters
to get augmentated data.**

Example of `augmentator_object` specified for augmentation:
```
augmentator_object_1 = Augmentator(noises_dataset=<path to dataset>,
                                   silero_vad_model_path=<path to silero_vad.jit>
                                   decibels=5.0,
                                   household_noises=True,
                                   background_music_noises=True) 
```
The parameters set of this example will let you get two files as output: original file augmented with household noises 
and music noises. The noise from corpora and augmentation way ('loop', 'random_position') is chosen randomly.

##### **1.2 Reverberation (`Augmentator(to_reverb=True)`)**

Reverberation parameters set:

* `to_reverb`: enable getting reverberation results (**always required for reverberation**), default = False;
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
augmentator_object_2 = Augmentator(to_reverb=True,
                                   wet_only=True) 
```

##### **2. Augmentation | reverberation outputs**

Use reverberate() or augmentate() to get outputs (see _simple_example.py_ and _mulitask_example.py_):

```
augmented_sound = augmentator_object_1.augmentate(audio_to_augment_input='wav_to_augment.wav', file_original_sample_rate=16000)
reverbed_sound = augmentator_object_2.reverberate(audio_to_reverb_input='wav_to_reverb.wav', file_original_sample_rate=16000)
augmented_sound  # {'<wav_to_augment>_household_noises_5.wav': <torch.tensor>, 
                    '<wav_to_augment>_background_music_noise_5.wav': <torch.tensor>}
reverbed_sound  # {'<wav_to_reverb>_reverbed.wav': <torch.tensor>}
```
**(!)** `audio_to_augment_input=` and `audio_to_reverb_input=` values can be got in the following format:
1) string path to audiofile;
2) torch.tensor;
3) numpy.ndarray.

**(!)** `file_original_sample_rate=` is 16000 by default, but it should be changes if it differs.

