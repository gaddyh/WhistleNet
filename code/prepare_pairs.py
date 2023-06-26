import os
import tensorflow_io as tfio
import librosa
import tensorflow as tf
import numpy as np
import itertools
import random
from keras.layers import dot
import matplotlib.pyplot as plt
from WhistleNet.code.augmentation import *
from WhistleNet.code.audio import *
from WhistleNet.code.audio2image import *

DEV_PATH = "WhistleNet/media/dev/"
VALIDATION_PATH = "WhistleNet/media/validation/"
TEST_PATH = "WhistleNet/media/test/"

MEDIA_PATH = "WhistleNet/media/"


def getall(media_dir):
    samplesbycategory = []
    i = 0
    for path in os.listdir(media_dir):
        samples_dir = os.path.join(media_dir, path)
        print(samples_dir)
        if not os.path.isfile(samples_dir):
            samplesbycategory.append(get_set(samples_dir))
            i = i + 1
    return samplesbycategory



def get_set(path):
    samples = get_samples(path)
    samples = augment_spec(samples)
    return samples


def get_samples(path):
    samples = []
    for file in os.listdir(path):
        # print(file)
        fp = os.path.join(path, file)
        samples = samples + load_max_samples(fp)
    return samples


def augment_spec(samples):
    signals = []
    for sample in samples:
        signal = np.array(sample)
        signal = np.squeeze(signal)
        # print(signal.shape,4)
        signals.append(signal)
        signals.append(noise(signal, noise_factor=0.005))
        signals.append(stretch(signal, 1.2))
        signals.append(stretch(np.array(signal), 0.78))
        signals.append(np.roll(signal, 300))

    #signals = list(map(get_mfcc2, signals))
    signals = list(map(get_mfcc2, signals))

    return signals


def split(samples_same_category, num_test):
    test = samples_same_category[:num_test]
    train = samples_same_category[num_test:]
    return train, test


def prnt(str):
    print("\n".join(str))


def create_ds_pairs(anchors1, negatives1):
    # when anchors only on left val[:,0] is bad val[:,1]] is excelent
    l = len(anchors1)
    lh = math.floor(l / 2)

    negatives = list(itertools.product(anchors1[:lh], negatives1))
    negatives = negatives + list(itertools.product(negatives1, anchors1[lh:]))
    anchors = list(itertools.combinations(anchors1, 2))

    X = anchors + negatives
    labels = (
        np.ones(len(anchors), dtype=int).tolist()
        + np.zeros((len(negatives),), dtype=int).tolist()
    )

    print(len(labels))

    X = np.array(X)
    labels = np.array(labels)

    p = [x for x in range(len(labels))]
    random.shuffle(p)

    X = X[p]
    labels = labels[p]

    return X, labels


def create_anchors_ds_pairs(anchors1, anchors2):
    # when anchors only on left val[:,0] is bad val[:,1]] is excelent
    l = len(anchors1)
    lh = math.floor(l / 2)

    negatives = list(itertools.product(anchors1[:lh], anchors2))
    negatives = negatives + list(itertools.product(anchors2, anchors1[lh:]))

    anchors = list(itertools.combinations(anchors1, 2))
    anchors = anchors + list(itertools.combinations(anchors2, 2))

    X = anchors + negatives
    labels = (
        np.ones(len(anchors), dtype=int).tolist()
        + np.zeros((len(negatives),), dtype=int).tolist()
    )

    print(len(labels))

    X = np.array(X)
    labels = np.array(labels)

    p = [x for x in range(len(labels))]
    random.shuffle(p)

    X = X[p]
    labels = labels[p]

    return X, labels


def create_anchors_list_pairs(anchors1, anchors2, anchors3):
    # when anchors only on left val[:,0] is bad val[:,1]] is excelent
    l = len(anchors1)
    lh = math.floor(l / 2)

    l2 = len(anchors2)
    lh2 = math.floor(l2 / 2)

    negatives = (
        list(itertools.product(anchors1[:lh], anchors2))
        + list(itertools.product(anchors2, anchors1[lh:]))
        + list(itertools.product(anchors1[:lh], anchors3))
        + list(itertools.product(anchors3, anchors1[lh:]))
        + list(itertools.product(anchors2[:lh2], anchors3))
        + list(itertools.product(anchors3, anchors2[lh2:]))
    )

    anchors = (
        list(itertools.combinations(anchors1, 2))
        + list(itertools.combinations(anchors2, 2))
        + list(itertools.combinations(anchors3, 2))
    )

    X = anchors + negatives
    labels = (
        np.ones(len(anchors), dtype=int).tolist()
        + np.zeros((len(negatives),), dtype=int).tolist()
    )

    return X, labels


def create_anchors_ds_pairs(anchors1, anchors2, anchors3):
    # when anchors only on left val[:,0] is bad val[:,1]] is excelent
    l = len(anchors1)
    lh = math.floor(l / 2)

    l2 = len(anchors2)
    lh2 = math.floor(l2 / 2)

    negatives = (
        list(itertools.product(anchors1[:lh], anchors2))
        + list(itertools.product(anchors2, anchors1[lh:]))
        + list(itertools.product(anchors1[:lh], anchors3))
        + list(itertools.product(anchors3, anchors1[lh:]))
        + list(itertools.product(anchors2[:lh2], anchors3))
        + list(itertools.product(anchors3, anchors2[lh2:]))
    )

    anchors = (
        list(itertools.combinations(anchors1, 2))
        + list(itertools.combinations(anchors2, 2))
        + list(itertools.combinations(anchors3, 2))
    )

    X = anchors + negatives
    labels = (
        np.ones(len(anchors), dtype=int).tolist()
        + np.zeros((len(negatives),), dtype=int).tolist()
    )

    print(len(labels))

    X = np.array(X)
    labels = np.array(labels)

    p = [x for x in range(len(labels))]
    random.shuffle(p)

    X = X[p]
    labels = labels[p]

    return X, labels


def create_anchors_ds_pairs(trainsbycategory):
    mids = list(range(len(trainsbycategory)))
    negatives = []
    anchors = []
    count = len(trainsbycategory) - 1
    for i in range(count):
        mids[i] = math.floor(len(trainsbycategory[i]) / 2)
        negatives = (
            negatives
            + list(
                itertools.product(
                    trainsbycategory[i][: mids[i]], trainsbycategory[i + 1]
                )
            )
            + list(
                itertools.product(
                    trainsbycategory[i + 1], trainsbycategory[i][mids[i] :]
                )
            )
        )
        anchors = anchors + list(itertools.combinations(trainsbycategory[i], 2))

    X = anchors + negatives
    labels = (
        np.ones(len(anchors), dtype=int).tolist()
        + np.zeros((len(negatives),), dtype=int).tolist()
    )

    print(len(labels))

    X = np.array(X)
    labels = np.array(labels)

    p = [x for x in range(len(labels))]
    random.shuffle(p)

    X = X[p]
    labels = labels[p]

    return X, labels

		
		
		# TODO
		# same samples per person. me has much more than others
		# each batch same positives and negatives
		
		
		
def create_all_train_pairs(samplesbycategory):
    negative_pairs = []
    positive_pairs = []
    for i,sc in enumerate(samplesbycategory):
        positive_pairs = positive_pairs + list(itertools.combinations(sc, 2))
    
    negatives = list(itertools.product(samplesbycategory[0], samplesbycategory[1])) + list(itertools.product(samplesbycategory[2], samplesbycategory[0])) + list(itertools.product(samplesbycategory[0], samplesbycategory[3])) + list(itertools.product(samplesbycategory[4], samplesbycategory[0], )) + list(itertools.product(samplesbycategory[0], samplesbycategory[5])) + list(itertools.product(samplesbycategory[1], samplesbycategory[2])) + list(itertools.product(samplesbycategory[3], samplesbycategory[1])) + list(itertools.product(samplesbycategory[1], samplesbycategory[4])) + list(itertools.product(samplesbycategory[5], samplesbycategory[1])) + list(itertools.product(samplesbycategory[2], samplesbycategory[3])) + list(itertools.product(samplesbycategory[4], samplesbycategory[2])) + list(itertools.product(samplesbycategory[2], samplesbycategory[5])) + list(itertools.product(samplesbycategory[3], samplesbycategory[4])) + list(itertools.product(samplesbycategory[4], samplesbycategory[3]))  + list(itertools.product(samplesbycategory[5], samplesbycategory[4]))

    X = positive_pairs + negatives
    labels = np.ones(len(positive_pairs), dtype=int).tolist() + np.zeros((len(negatives),), dtype=int).tolist()

    return X, labels

def create_hard_pairs(samplesbycategory, model, semi=True):
    hard_negative_pairs = []
    hard_positive_pairs = []
    for i in range(len(samplesbycategory)):
        samples_np = np.array(samplesbycategory[i])
        representations_i = model.predict(samples_np)
        if semi:
            pairs = list(itertools.combinations(samples_np, 2))
            hard_positive_pairs = hard_positive_pairs + pairs
        else:
            pairs = list(itertools.combinations(representations_i, 2))
            dots = []
            for pair in pairs:
                a, b = pair
                dot_product = dot([[a], [b]], axes=1, normalize=True)
                if dot_product < 0.5:
                    resulta = np.where((representations_i == a).all(axis=1))[0]
                    resultb = np.where((representations_i == b).all(axis=1))[0]
                    hard_pairs = (samples_np[resulta][0], samples_np[resultb][0])
                    hard_positive_pairs.append(hard_pairs)

        for j in range(len(samplesbycategory)):
            if i != j:
                samples_np1 = np.array(samplesbycategory[j])
                representations_j = model.predict(samples_np1)
                pairs = list(itertools.product(representations_i, representations_j))
                dots = []
                for pair in pairs:
                    a, b = pair
                    resulta = np.where((representations_i == a).all(axis=1))[0]
                    resultb = np.where((representations_j == b).all(axis=1))[0]
                    hard_pairs = (samples_np[resulta][0], samples_np1[resultb][0])
                    hard_negative_pairs.append(hard_pairs)

        X = hard_positive_pairs + hard_negative_pairs
        labels = (
            np.ones(len(hard_positive_pairs), dtype=int).tolist()
            + np.zeros((len(hard_negative_pairs),), dtype=int).tolist()
        )

    return X, labels
