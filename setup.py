from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess


class PostInstallCommand(install):
    def run(self):
        subprocess.call(['pip', 'install', '-r', 'requirements.txt'])
        subprocess.call(['python', './audio_augmentator/load_files.py'])
        install.run(self)


setup(
    name='audio_augmentator',
    version='0.3.2',
    description='a Python library for audio augmentation',
    url='https://github.com/dangrebenkin/audio_augmentator',
    author='Daniel Grebenkin',
    author_email='d.grebenkin@g.nsu.ru',
    license='Apache License Version 2.0',
    keywords=['wav', 'audio', 'augmentation', 'reverberation'],
    packages=find_packages(),
    python_requires=r'>=3.8.0',
    cmdclass={
        'install': PostInstallCommand,
    },
)
