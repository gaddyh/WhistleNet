import os
ANCOR_PATH = 'WhistleNet/media/anchor/'
NEGATIVE_PATH = 'WhistleNet/media/negative/'
import tensorflow_io as tfio
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt


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

def audio2image(full_path):
  if full_path in ready:
    print('ready')
    return ready[full_path]
 
  print(full_path)
  
  isExist = os.path.exists(full_path) # original, no 
  if(isExist):
    #audio = tfio.audio.AudioIOTensor(full_path)
    audio, s = librosa.load(full_path, sr=rate) # Downsample to 44.1kHz

    print(audio.shape) 
    audio_slice = audio[:rate] # one second, maybe trim silence before or split to 2 if enough data
    #audio_slice = tf.slice(audio,begin=[0], size=[47000])
    # remove last dimension
    # audio_tensor = tf.squeeze(audio_slice, axis=[1])
    #print(audio_tensor) 
    tensor = tf.cast(audio_slice, tf.float32) / 32768.0

    spectrogram = tfio.audio.spectrogram(
    tensor, nfft=512, window=512, stride=256)
    
    #plt.figure()
    #plt.imshow(tf.math.log(spectrogram).numpy())

    mel_spectrogram = tfio.audio.melscale(
    spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
    
    #plt.figure()
    #plt.imshow(tf.math.log(mel_spectrogram).numpy())

    dbscale_mel_spectrogram = tfio.audio.dbscale(
    mel_spectrogram, top_db=80)
    plt.figure()
    plt.imshow(dbscale_mel_spectrogram.numpy())
    plt.savefig(os.path.splitext(full_path)[0] + '.png')
    ready[full_path] =  plt.imread(os.path.splitext(full_path)[0] + '.png')
    return ready[full_path]


def getall(anchorsdir):
  anchorfns = []
  for path in os.listdir(anchorsdir):
    full_path = os.path.join(anchorsdir, path)
    ext = os.path.splitext(full_path)[1]
    if ext != ".png" and os.path.isfile(full_path):
      anchorfns.append(full_path)
      anchorfns.append(anchorsdir + os.path.splitext(path)[0] + '_noise.wav')
      anchorfns.append(anchorsdir + os.path.splitext(path)[0] + '_roll.wav')
      anchorfns.append(anchorsdir + os.path.splitext(path)[0] + '_strech08.wav')
      anchorfns.append(anchorsdir + os.path.splitext(path)[0] + '_strech12.wav')
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
