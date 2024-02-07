import random

import pytest
import torch

from audio_augmentator import Augmentator

NOISES_DATASET_PATH = r'C:\Users\LimpWinter\Documents\Projects\audio_augmentator\noises_dataset_2'


def test_augmenter_init():
    augmenter = Augmentator(
        noises_dataset=NOISES_DATASET_PATH,
        decibels=10.0
    )
    assert augmenter is not None

    assert augmenter.noises_dataset is not None
    assert augmenter.decibels == 10.0
    assert augmenter.household_noises is False
    assert augmenter.pets_noises is False
    assert augmenter.speech_noises is False
    assert augmenter.background_music_noises is False


def test_augmenter_gives_all_noises():
    augmenter = Augmentator(
        noises_dataset=NOISES_DATASET_PATH,
        decibels=10.0,
        household_noises=True,
        pets_noises=True,
        speech_noises=True,
        background_music_noises=True,
        to_mix=True
    )

    sample = torch.randn(16_000)
    augmented_audio = augmenter.augmentate(sample)
    assert isinstance(augmented_audio, dict), "Augmenter should return a dictionary"

    assert len(augmented_audio) == 5, "Augmenter should return 4 types of noises and 1 mixed noise"


def test_augmenter_gives_no_noises():
    augmenter = Augmentator(
        noises_dataset=NOISES_DATASET_PATH,
        decibels=10.0
    )
    sample = torch.randn((1, 16_000))
    augmented_audio = augmenter.augmentate(sample)
    assert isinstance(augmented_audio, dict), "Augmenter should return a dictionary"
    assert len(augmented_audio) == 0, "Augmenter with no arguments should return an empty dictionary"


def test_augmenter_keeps_audio_shape():
    augmenter = Augmentator(
        noises_dataset=NOISES_DATASET_PATH,
        decibels=10.0,
        household_noises=True,
        pets_noises=True,
        speech_noises=True,
        background_music_noises=True,
        to_mix=True
    )
    for i in range(10):
        sample = torch.randn((1, random.randint(2_000, 160_000)))
        augmented_audio = augmenter.augmentate(sample)

        for aug_sample in augmented_audio.keys():
            assert augmented_audio[aug_sample].shape == sample.shape, \
                "Augmentator should return audio with the same shape as input audio"


def test_augmenter_changes_audio():
    augmenter = Augmentator(
        noises_dataset=NOISES_DATASET_PATH,
        decibels=5.0,
        household_noises=True,
        pets_noises=True,
        speech_noises=True,
        background_music_noises=True,
        to_mix=True
    )
    for i in range(10):
        sample = torch.randn((1, random.randint(2_000, 160_000)))
        augmented_audio = augmenter.augmentate(sample)

        for aug_sample in augmented_audio.keys():
            assert not torch.norm(augmented_audio[aug_sample] - sample, p='fro') < 1, \
                "Augmentator should change audio"


def test_augmenter_household_noises():
    augmenter = Augmentator(
        noises_dataset=NOISES_DATASET_PATH,
        decibels=5.0,
        household_noises=True,
    )
    for i in range(10):
        sample = torch.randn((1, random.randint(2_000, 160_000)))
        augmented_dict = augmenter.augmentate(sample)

        assert isinstance(augmented_dict, dict), "Augmenter should return a dictionary"
        assert len(augmented_dict) == 1, "Augmenter should return only one household noise"

        for aug_sample_name in augmented_dict.keys():
            assert "household" in aug_sample_name, \
                "Augmenter should return only household noises"


def test_augmenter_pets_noises():
    augmenter = Augmentator(
        noises_dataset=NOISES_DATASET_PATH,
        decibels=5.0,
        pets_noises=True,
    )
    for i in range(10):
        sample = torch.randn((1, random.randint(2_000, 160_000)))
        augmented_dict = augmenter.augmentate(sample)

        assert isinstance(augmented_dict, dict), "Augmenter should return a dictionary"
        assert len(augmented_dict) == 1, "Augmenter should return only one pets noise"

        for aug_sample_name in augmented_dict.keys():
            assert "pets" in aug_sample_name, \
                "Augmenter should return only pets noises"


def test_augmenter_speech_noises():
    augmenter = Augmentator(
        noises_dataset=NOISES_DATASET_PATH,
        decibels=5.0,
        speech_noises=True,
    )
    for i in range(10):
        sample = torch.randn((1, random.randint(2_000, 160_000)))
        augmented_dict = augmenter.augmentate(sample)

        assert isinstance(augmented_dict, dict), "Augmenter should return a dictionary"
        assert len(augmented_dict) == 1, "Augmenter should return only one speech noise"

        for aug_sample_name in augmented_dict.keys():
            assert "speech" in aug_sample_name, \
                "Augmenter should return only speech noises"


def test_augmenter_background_music_noises():
    augmenter = Augmentator(
        noises_dataset=NOISES_DATASET_PATH,
        decibels=5.0,
        background_music_noises=True,
    )
    for i in range(10):
        sample = torch.randn((1, random.randint(2_000, 160_000)))
        augmented_dict = augmenter.augmentate(sample)

        assert isinstance(augmented_dict, dict), "Augmenter should return a dictionary"
        assert len(augmented_dict) == 1, "Augmenter should return only one background music noise"

        for aug_sample_name in augmented_dict.keys():
            assert "background_music" in aug_sample_name, \
                "Augmenter should return only background music noises"


def test_no_noises_to_mix():
    with pytest.raises(AssertionError):
        _ = Augmentator(
            noises_dataset=NOISES_DATASET_PATH,
            decibels=5.0,
            to_mix=True
        )
    with pytest.raises(AssertionError):
        _ = Augmentator(
            noises_dataset=NOISES_DATASET_PATH,
            decibels=5.0,
            household_noises=True,
            to_mix=True
        )
    with pytest.raises(AssertionError):
        _ = Augmentator(
            noises_dataset=NOISES_DATASET_PATH,
            decibels=5.0,
            speech_noises=True,
            to_mix=True
        )
