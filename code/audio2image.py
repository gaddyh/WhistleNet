import tensorflow as tf
import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
sample_rate = 48000

def get_mfcc2(signal, sr=48000):
  mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20,lifter=100)
  return mfccs


def get_mfcc22323(signal):
# A 1024-point STFT with frames of 64 ms and 75% overlap.
  stfts = tf.signal.stft(signal, frame_length=1024, frame_step=256,
                         fft_length=1024)
  spectrograms = tf.abs(stfts)

  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = 513
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
  mel_spectrograms = tf.tensordot(
    spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :50]
  
  return mfccs

 
 # look at more than one mfcc !! do they have equal shapes/sizes? fix size. find min and max sizes for all dataset
 #understand this:
 #https://www.kaggle.com/code/ilyamich/mfcc-implementation-and-tutorial
def get_mfcc(waveform, sample_rate=48000):
  audio = normalize_audio(waveform)
  hop_size = 15 #ms
  FFT_size = 2048

  audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
  
  window = get_window("hann", FFT_size, fftbins=True)
  audio_win = audio_framed * window
  audio_winT = np.transpose(audio_win)

  audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

  for n in range(audio_fft.shape[1]):
    audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]

  audio_fft = np.transpose(audio_fft)
  
  audio_power = np.square(np.abs(audio_fft))
  
  freq_min = 0
  freq_high = sample_rate / 2
  mel_filter_num = 20
  
  filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=48000)

  filters = get_filters(filter_points, FFT_size)
  
  enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
  filters *= enorm[:, np.newaxis]
  
  audio_filtered = np.dot(filters, np.transpose(audio_power))
  audio_log = 10.0 * np.log10(audio_filtered)
  
  dct_filter_num = 40

  dct_filters = dct(dct_filter_num, mel_filter_num)

  cepstral_coefficents = np.dot(dct_filters, audio_log)
  
  return cepstral_coefficents
  
def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis  
  
def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters  
  
def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=48000):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    print("MEL min: {0}".format(fmin_mel))
    print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs
  
def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=48000):
    # hop_size in ms
    
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
    
    return frames
    
def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram
  
  
