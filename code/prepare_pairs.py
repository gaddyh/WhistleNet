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

MEDIA_PATH = 'WhistleNet/media/'

def getall(media_dir):
	samplesbycategory = []
	i=0
	for path in os.listdir(media_dir):
		samples_dir = os.path.join(media_dir, path)
		samplesbycategory.append(get_set(samples_dir))
		i=i+1
	return samplesbycategory

	
def get_set(path):
	samples = get_samples(path)
	samples = augment_spec(samples)
	return samples
	
def get_samples(path):
	samples=[]
	for file in os.listdir(path):
			#print(file)
			fp = os.path.join(path, file)
			samples=samples+load_max_samples(fp)
	return samples

def augment_spec(samples):
	signals=[]
	for sample in samples:
		signal = np.array(sample)
		signal = np.squeeze(signal)
		#print(signal.shape,4)
		signals.append(signal)
		signals.append(noise(signal, noise_factor = 0.005 ))
		#signals.append(stretch(signal, 1.2))
		#signals.append(stretch(np.array(signal), 0.78))
		signals.append(np.roll(signal, 300))

	signals = list(map(get_spectrogram, signals))

	return signals

def split(samples_same_category, num_test):
	test = samples_same_category[:num_test]
	train = samples_same_category[num_test:]
	return train, test
	
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
	
def create_anchors_ds_pairs(anchors1, anchors2):
	#when anchors only on left val[:,0] is bad val[:,1]] is excelent
	l = len(anchors1)
	lh = math.floor(l/2)

	negatives = list(itertools.product(anchors1[:lh], anchors2))
	negatives = negatives + list(itertools.product(anchors2, anchors1[lh:]))
	
	anchors = list(itertools.combinations(anchors1,2))
	anchors = anchors + list(itertools.combinations(anchors2,2))

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

def create_anchors_list_pairs(anchors1, anchors2, anchors3):
	#when anchors only on left val[:,0] is bad val[:,1]] is excelent
	l = len(anchors1)
	lh = math.floor(l/2)

	l2 = len(anchors2)
	lh2 = math.floor(l2/2)
	
	negatives = list(itertools.product(anchors1[:lh], anchors2)) + list(itertools.product(anchors2, anchors1[lh:])) + list(itertools.product(anchors1[:lh], anchors3)) + list(itertools.product(anchors3, anchors1[lh:])) + list(itertools.product(anchors2[:lh2], anchors3)) + list(itertools.product(anchors3, anchors2[lh2:]))
							
	
	anchors = list(itertools.combinations(anchors1,2)) + list(itertools.combinations(anchors2,2)) + list(itertools.combinations(anchors3,2))

	X = anchors + negatives
	labels = np.ones(len(anchors), dtype=int).tolist() + np.zeros((len(negatives),), dtype=int).tolist()
	
	return X, labels


def create_anchors_ds_pairs(anchors1, anchors2, anchors3):
	#when anchors only on left val[:,0] is bad val[:,1]] is excelent
	l = len(anchors1)
	lh = math.floor(l/2)

	l2 = len(anchors2)
	lh2 = math.floor(l2/2)
	
	negatives = list(itertools.product(anchors1[:lh], anchors2)) + list(itertools.product(anchors2, anchors1[lh:])) + list(itertools.product(anchors1[:lh], anchors3)) + list(itertools.product(anchors3, anchors1[lh:])) + list(itertools.product(anchors2[:lh2], anchors3)) + list(itertools.product(anchors3, anchors2[lh2:]))
							
	
	anchors = list(itertools.combinations(anchors1,2)) + list(itertools.combinations(anchors2,2)) + list(itertools.combinations(anchors3,2))

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


def create_anchors_ds_pairs(trainsbycategory):
  mids = []
  negatives = []
  for i in len(trainsbycategory):
    mids[i] = math.floor(len(trainsbycategory[i])/2)
	
    negatives = list(itertools.product(anchors1[:lh], anchors2)) + list(itertools.product(anchors2, anchors1[lh:])) + list(itertools.product(anchors1[:lh], anchors3)) + list(itertools.product(anchors3, anchors1[lh:])) + list(itertools.product(anchors2[:lh2], anchors3)) + list(itertools.product(anchors3, anchors2[lh2:]))
							
	
  anchors = list(itertools.combinations(anchors1,2)) + list(itertools.combinations(anchors2,2)) + list(itertools.combinations(anchors3,2))

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
  
def create_hard_pairs(samplesbycategory, model):
  hard_negative_pairs = []
  hard_positive_pairs = []
  for i in range(len(samplesbycategory)):
    samples_np = np.array(samplesbycategory[i])
    representations_i = model.predict(samples_np)
    pairs = list(itertools.combinations(representations_i,2))
    dots=[]
    for pair in pairs:
      a,b = pair
      dot_product = a@b
      if(dot_product < 0.5):
        resulta = np.where((representations_i == a).all(axis=1))[0]
        resultb = np.where((representations_i == b).all(axis=1))[0]
        hard_pairs = (samples_np[resulta][0], samples_np[resultb][0])
        hard_positive_pairs.append(hard_pairs)

    for j in range(len(samplesbycategory)):
      if i != j:
        samples_np1 = np.array(samplesbycategory[j])
        representations_j = model.predict(samples_np1)
        pairs = list(itertools.product(representations_i, representations_j))
        dots=[]
        for pair in pairs:
          a,b = pair
          dot_product = a@b
          if(dot_product > 0.5):
            resulta = np.where((representations_i == a).all(axis=1))[0]
            resultb = np.where((representations_j == b).all(axis=1))[0]
            hard_pairs = (samples_np[resulta][0], samples_np1[resultb][0])
            hard_negative_pairs.append(hard_pairs)

  X = hard_positive_pairs + hard_negative_pairs
  labels = np.ones(len(hard_positive_pairs), dtype=int).tolist() + np.zeros((len(hard_negative_pairs),), dtype=int).tolist()
  return X, labels
