## Аугментатор аудиофайлов

Библиотека для аугментации аудиозаписей в формате моно WAV PCM, частота дискретизации 8 или 16 кГц.

Аугментатор способен создавать зашумленные аудиозаписи с эффектом реверберации или с фоновым шумом:

1. Для получения эффекта реверберации используется
   библиотека [pysndfx](https://github.com/carlthome/python-audio-effects)
   c настраиваемыми параметрами глубины, уменьшения высоких частот и т.п.
2. Корпус доступных фоновых шумов содержит 4 типа аудиозаписей:
    - Бытовые (стук столовых приборов, шум работающей стиральной машинки хлопок межкомнатной двери и др.);
    - Шумы домашних животных (мяуканье кошек, лай собак и др.);
    - Речеподобные шумы (фоновая речь других людей, шум телевизора или радио и др.);
    - Музыкальные шумы (фоновая музыка и др.).

Корпус шумов был сформирован из нескольких источников:

- [Animal-Sound-Dataset](https://github.com/YashNita/Animal-Sound-Dataset);
- [Freesound Audio Tagging 2019](https://www.kaggle.com/competitions/freesound-audio-tagging-2019/data).

Шумовые аудиозаписи могут быть «подмешаны» в различных сочетаниях с настраиваемым значением соотношения сигнал-шум
(в децибелах) одним из двух способов, который выбирается случайно (с помощью инструментов библиотеки random):

1. Если в результате случайного выбора избирается способ «цикла» («loop»), то вычисляется число необходимых
   повторений шумового аудиосигнала/аудиофрагмента, которое должно быть не меньше чем входной аудиосигнал по
   длительности.
   После этого создается новый шумовой аудиосигнал путем конкатенации вычисленного числа шумовых
   аудиосигналов/аудиофрагментов;
2. Если в результате случайного выбора избирается способ «случайной позиции» («random position»), то случайным
   образом вычисляется временной момент начала и конца «наложения» шума с учетом длительности входного
   аудиосигнала;
3. Далее в каждом из способов шумовой аудиосигнал «накладывается» на входной аудиосигнал с учетом его
   длительности
   (новый шумовой аудиосигнал обрезается по моменту окончания входного аудиосигнала) и выбранного пользователем
   соотношения сигнал-шум.

### Описание предобработки датасета шумов

В процессе формирования корпуса шумовых аудиозаписей файлы длительностью менее 3 секунд сохранялись для участия
в аугментации целиком; если шумовая аудиозапись имела длительность более 3 секунд, то из такого сигнала несколькими
методами в зависимости от типа шума выделяется наиболее подходящий для аугментации фрагмент:

- Для речеподобных шумов использовалась модель [Silero Voice Active Detection](https://github.com/snakers4/silero-vad);
- Для остальных типов шумов был разработан и реализован следующий алгоритм:

    1. Выполняется поиск самого большого по значению локального максимума функции энергии аудиосигнала;
    2. В окрестностях найденного локального максимума выполняется поиск локальных минимумов значений функции энергии,
       которые и определяют границы конечного аудиофрагмента.
       После окончания подготовки шумового аудиофрагмента или аудиофайла целиком шумовая аудиозапись может быть
       «подмешана» во входную аудиозапись 

### Установка

```
sudo apt-get install sox
git clone https://github.com/dangrebenkin/audio_augmentator
cd audio_augmentator
python setup.py install
```

**При установке автоматически скачивается предобработанный корпус шумов.**

### Тестирование 
Для базового тестирования используется бибилиотека `pytest` и следующая команда:
```bash
python -m pytest tests/
```

### Алгоритм использования

#### **1. Создать экземпляр класса _Augmentator_ с необходимыми параметрами:**

```
from audio_augmentator import Augmentator

augmentator_object = Augmentator(
    noises_dataset=<path to dataset>,
    ...
)
```
Настраиваемые параметры:

**Обязательный параметр (всегда):**

* `noises_dataset` : полный путь к датасету шумов _noises_dataset_, который скачивается автоматически при установке
  библиотеки;

**Список дополнительных параметров:**

* `decibels`: соотношение сигнал-шум (в дБ), (**опционально**) по умолчанию = 10.0;
* `household_noises`: установить True для получения аудио, аугментированных бытовыми шумами (**опционально**), по
  умолчанию = False;
* `pets_noises`: установить True для получения аудио, аугментированных шумами животных (**опционально**), по умолчанию =
  False;
* `speech_noises`: установить True для получения аудио, аугментированных речеподобными шумами (**опционально**), по
  умолчанию = False;
* `background_music_noises`: установить True для получения аудио, аугментированных музыкальными шумами (**опционально
  **), по умолчанию = False;
* `to_mix`: установить True для получения аудио, аугментированных несколькими типами шумамов (для срабатывания
  необходимо, чтобы >= 2 типов шумов (`household_noises`, `pets_noises`, `speech_noises`, `background_music_noises`)
  были установлены в True) (
  **опционально**), по умолчанию = False;
* `to_reverb`: установить True для получения аудио, аугментированных ревербационными шумами (**always required for
  reverberation**), по умолчанию = False.

**Также доступно изменения параметров из библиотеки pysndfx для настройки реверберации:**

* `reverberance`: (**опционально**) по умолчанию = 50;
* `hf_damping`: (**опционально**) по умолчанию = 50;
* `room_scale`: (**опционально**) по умолчанию = 100;
* `stereo_depth`: (**опционально**) по умолчанию = 100;
* `pre_delay`: (**опционально**) по умолчанию = 20;
* `wet_gain`: (**опционально**) по умолчанию = 0;
* `wet_only`: (**опционально**) по умолчанию = False).

Примеры:

1. Инициализация объекта класса _Augmentator_ для аугментации:

```
from audio_augmentator import Augmentator

augmentator_object_1 = Augmentator(
    noises_dataset=noises_dataset,
    decibels=5.0,
    speech_noises=True,
    background_music_noises=True
)
```

2. Инициализация объекта класса _Augmentator_ для реверберации:

```
from audio_augmentator import Augmentator

augmentator_object_2 = Augmentator(
    noises_dataset=noises_dataset,
    to_reverb=True,
    room_scale=80
)
```

3. Инициализация объекта класса _Augmentator_ для реверберации и аугментации:

```
from audio_augmentator import Augmentator

augmentator_object_3 = Augmentator(
    noises_dataset=noises_dataset,
    to_reverb=True,
    room_scale=80,
    decibels=15.0,
    background_music_noises=True
)
```

#### **2. Применить функцию reverberate() и/или augmentate() к аудио**

```
augmented_sound = augmentator_object_1.augmentate(audio_to_augment_input='wav_to_augment.wav', 
                                                  file_original_sample_rate=16000)
reverbed_sound = augmentator_object_2.reverberate(audio_to_reverb_input='wav_to_reverb.wav', 
                                                  file_original_sample_rate=16000)
augmented_sound  # {'<wav_to_augment>_household_noises_5.wav': <torch.tensor>, 
                    '<wav_to_augment>_background_music_noise_5.wav': <torch.tensor>}
reverbed_sound  # {'<wav_to_reverb>_reverbed.wav': <torch.tensor>}
```

**(!)** В качестве входного аудио (параметры `audio_to_augment_input=` and `audio_to_reverb_input=`) можно подать данные
следующих типов:

1) string path to audiofile;
2) torch.Tensor;
3) numpy.Ndarray.

**(!)** Параметр `file_original_sample_rate=` (**опционально**) 16000 по умолчанию, но он может быть изменен в
соответствии с оригинальной частотой дискретизации входного аудио файла.

**(!)** Больше примеров использования аугментатора можно найти в папке `examples`.

### Ссылки

1. Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, Xavier Serra. Audio tagging with noisy labels and
   minimal supervision. In Proceedings of DCASE2019 Workshop, NYC, US (2019). URL: https://arxiv.org/abs/1906.02975
2. @misc{Silero VAD,
   author = {Silero Team},
   title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language
   Classifier},
   year = {2021},
   publisher = {GitHub},
   journal = {GitHub repository},
   howpublished = {\url{https://github.com/snakers4/silero-vad}},
   commit = {insert_some_commit_here},
   email = {hello@silero.ai}
   }

### Благодарности

[Артему Болдинову aka LimpWinter](https://github.com/limpwinter) за рефакторинг, тестирование и 
подготовку новой версии библиотеки.

