import os
import tensorflow_io as tfio
import librosa
import tensorflow as tf
import numpy as np
import itertools
import random

import matplotlib.pyplot as plt
from WhistleNet.code.augmentation import *
from WhistleNet.code.audio import *
from WhistleNet.code.audio2image import *

DEV_PATH = 'WhistleNet/media/dev/'
VALIDATION_PATH = 'WhistleNet/media/validation/'
TEST_PATH = 'WhistleNet/media/test/'

def get_set(path):
	anchors = get_samples(path + 'anchor/')
	negatives = get_samples(path + 'negative/')
	
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

	return signals

def getall(anchorsdir):
  anchorfns = []
  for path in os.listdir(anchorsdir):
    full_path = os.path.join(anchorsdir, path)
    ext = os.path.splitext(full_path)[1]
    if ext != ".png" and os.path.isfile(full_path):
      anchorfns.append(full_path)
  return anchorfns

def prnt(str):
  print("\n".join(str))

  
def create_ds_pairs(anchors1, negatives1):
  #when anchors only on left val[:,0] is bad val[:,1]] is excelent
  l = len(anchors1)
  lh = math.floor(l/2)

  negatives = list(itertools.product(anchors1[:lh], negatives1))
  negatives = negatives + list(itertools.product(negatives1, anchors1[lh:]))
  anchors = list(itertools.combinations(anchors1,2))

  X = anchors + negatives
  labels = np.ones(len(anchors), dtype=int).tolist() + np.zeros((len(negatives),), dtype=int).tolist()

  print(len(labels))

  X = np.array(X)
  labels = np.array(labels)

  p = [x for x in range(len(labels))]
  random.shuffle(p)

  X = X[p]
  labels = labels[p]

  return X, labels



model, siamese_model = create_siamese_model()

#print(model.summary())

history = siamese_model.fit([X[:,0], X[:,1]], labels, epochs=30, batch_size=32,
                            validation_data=([val[:,0], val[:,1]], val_labels),
                            shuffle=False, verbose=True)

#len(anchors), len(negatives),len(X),
print('--------------')
#X[:,0].shape,X[:,1].shape, labels.shape