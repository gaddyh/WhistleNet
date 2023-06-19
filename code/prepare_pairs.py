import os
import tensorflow_io as tfio
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from augmentation import *

DEV_PATH = 'WhistleNet/media/dev/'
VALIDATION_PATH = 'WhistleNet/media/validation/'
TEST_PATH = 'WhistleNet/media/test/'

def get_set(path):
	anchors = get_samples(path + 'anchor/')
	negatives = get_samples(path + '/negative/')
	
	anchors = augment_spec(anchors)
	negatives = augment_spec(negatives)
	
	return anchors, negatives


def get_samples(path):
	samples=[]
	files = getall(path)
	for file in files:
	  samples=samples+load_max_samples(file)
	return samples

def augment_spec(samples):
	signals=[]
	for sample in samples:
	  signal = np.array(sample)
	  signal = np.squeeze(signal)
	  #print(signal.shape,4)
	  signals.append(signal)
	  signals.append(noise(signal, noise_factor = 0.005 ))
	  signals.append(stretch(signal, 1.2))
	  signals.append(stretch(np.array(signal), 0.78))
	  signals.append(np.roll(signal, 300))

	signals = list(map(get_spectrogram, signals))








def loadpairs(anchorfns, negativefns):
  color_X1 = []
  color_X2 = []
  color_y = []

  for fname in anchorfns :
    print(fname[0])
    im = audio2image(fname[0])
    color_X1.append(im)
    im = audio2image(fname[1])
    color_X2.append(im)
    color_y.append(1)

  for fname in negativefns :
    im = audio2image(fname[0])
    color_X1.append(im)
    im = audio2image(fname[1])
    color_X2.append(im)
    color_y.append(0)

  return color_X1, color_X2, color_y

ready={"a": 1, "b":2, "c":3} # dummy dic
rate=44100

def getall(anchorsdir):
  anchorfns = []
  for path in os.listdir(anchorsdir):
    full_path = os.path.join(anchorsdir, path)
    ext = os.path.splitext(full_path)[1]
    if ext != ".png" and os.path.isfile(full_path):
      anchorfns.append(full_path)
  return anchorfns


def test2(path):
  print('test2')
  return path

def mediafiles():
  media=[]
  for file in os.listdir(ANCOR_PATH):
    media.append(ANCOR_PATH + file)

  for file in os.listdir(NEGATIVE_PATH):
    media.append(NEGATIVE_PATH + file)

  print("\n".join(media))
  return media

def prnt(str):
  print("\n".join(str))

def printmediafiles(media):
  for file in media:  
    audio = tfio.audio.AudioIOTensor(file)
    print(file)
    print(audio)
