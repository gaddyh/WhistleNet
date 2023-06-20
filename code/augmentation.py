sample_rate = 48000

# must be in main/colab
#!pip install audio2numpy

import librosa
import numpy as np

def stretch(data, rate=1):
	input_length = sample_rate
	data = librosa.effects.time_stretch(data, rate=rate)
	if len(data)>input_length:
		data = data[:input_length]
	else:
		data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

	return data
	
def add_noise(original_audio):
  noise_factor = 0.005
  white_noise = np.random.randn(len(original_audio)) * noise_factor
  return original_audio + white_noise
  
def add_roll(original_audio):
	return  np.roll(original_audio, 3000)
	
def noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data
	
def shift(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif self.shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data
	
def pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

	
