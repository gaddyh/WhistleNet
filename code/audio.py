sample_rate = 44100

import numpy as np
import tensorflow as tf

# must be in main/colab
#!pip install audio2numpy

from audio2numpy import open_audio

def load(path):
	signal, sampling_rate = open_audio(path)
	signal = np.pad(signal, (0, max(0, sample_rate - len(signal))), "constant")
	return signal
	
def load_tensor(path, sample_rate):
  x = tf.io.read_file(str(path))
  waveform, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=sample_rate,)
  return waveform
  
import math
def load_max_samples(path):
  samples = []

  signal, sampling_rate = open_audio(path)
  if(sampling_rate == sample_rate):
    if len(signal) < sample_rate :
      signal = np.pad(signal, (0, max(0, sample_rate - len(signal))), "constant")
      samples.append(signal)
    else:
      count = math.floor(len(signal)/sample_rate)
      for i in range(count):
		if i == 0:
			samples.append( signal[:sample_rate])
		else:
			start = sample_rate * i
			end = sample_rate * (i+1)
			samples.append( signal[start:end])
  else:
    print('bad sample rate for ' , path, 'sample is: ', sampling_rate)
  
  return samples

def load_max_samples1(path):
  samples = []

  signal, sampling_rate = open_audio(path)
  if(sampling_rate == sample_rate):
    if len(signal) < sample_rate :
      signal = np.pad(signal, (0, max(0, sample_rate - len(signal))), "constant")
      samples.append(signal)
    elif len(signal) < 1.8*sample_rate :
      signal = signal[:sample_rate]
      samples.append(signal)
    elif(len(signal) < 2.8*sample_rate ): # 1.8 < signal < 2.8
      out = np.array_split(signal, 2)
      signal1 = np.pad(out[0], (0, max(0, sample_rate - len(out[0]))), "constant")
      signal2 = np.pad(out[1], (0, max(0, sample_rate - len(out[1]))), "constant")
      signal1 = signal[:sample_rate]
      signal2 = signal[:sample_rate]
      samples.append(signal1)
      samples.append(signal2)

    elif(len(signal) < 3.8*sample_rate ): # 2.8 < signal < 3.8
      out = np.array_split(signal, 3)
      signal1 = np.pad(out[0], (0, max(0, sample_rate - len(out[0]))), "constant")
      signal2 = np.pad(out[1], (0, max(0, sample_rate - len(out[1]))), "constant")
      signal3 = np.pad(out[2], (0, max(0, sample_rate - len(out[2]))), "constant")
      signal1 = signal[:sample_rate]
      signal2 = signal[:sample_rate]
      signal3 = signal[:sample_rate]
      samples.append(signal1)
      samples.append(signal2)
      samples.append(signal3)

  else:
    print('bad sample rate for ' , path, 'sample is: ', sampling_rate)
  
  return samples