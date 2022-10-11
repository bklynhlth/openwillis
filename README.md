# openwillis

digital measurement of health

# Installation
1) Create environment:

```shell
conda create -n openwillis_env python=3.8
conda activate openwillis_env
```

2) Install `FFmpeg`:

```shell
sudo apt-get install ffmpeg
```

3) Install `PortAudio` and `soundfile`:

```shell
conda install portaudio
conda install pysoundfile -c conda-forge
```

4) Install openwillis:

```shell
pip install openwillis
```
