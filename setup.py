from setuptools import setup, find_packages

setup(
    name='audio_augmentator',
    version='0.1',
    description='a Python library for audio augmentation',
    url='https://github.com/dangrebenkin/audio_separator',
    author='Daniel Grebenkin',
    author_email='d.grebenkin@g.nsu.ru',
    license='Apache License Version 2.0',
    keywords=['wav', 'audio', 'augmetation', 'reverberation'],
    packages=find_packages(),
    python_requires=r'>=3.8.0',
)
