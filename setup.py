from setuptools import setup


def _get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements


setup(
    name='speechemotionrecognition',
    version='1.0',
    packages=['speechemotionrecognition'],
    url='https://github.com/nijatmursali/NeuralNetworks',
    license='MIT',
    install_requires=_get_requirements(),
    author='NijatMursali',
    author_email='nmursali2019@ada.edu.az',
    description='Speech Emotion Recognition using LSTM'
)
