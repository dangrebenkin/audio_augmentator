from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess


class PostInstallCommand(install):
    def run(self):
        subprocess.call(['pip', 'install', '-r', 'requirements.txt'])
        install.run(self)


setup(
    name='audio_augmentator',
    version='0.5',
    description='A Python library for audio augmentation and reverberation.',
    url='https://github.com/dangrebenkin/audio_augmentator',
    author='Daniel Grebenkin',
    author_email='d.grebenkin@g.nsu.ru',
    license='Apache License Version 2.0',
    keywords=['wav', 'audio', 'augmentation', 'reverberation'],
    packages=find_packages(),
    python_requires=r'>=3.10',
    cmdclass={
        'install': PostInstallCommand,
    },
)
