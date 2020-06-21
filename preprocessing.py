import matplotlib.pyplot as plt
from pathlib import Path
import os
from keras.preprocessing.image import ImageDataGenerator


def check_dir(path):
  if not path.exists():
    path.mkdir()

TARGET_SIZE=(128,128)
BATCH_SIZE=128


def preprocess_vol(path):
  seed = 1
  datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 5)
  generator = datagen.flow_from_directory(directory = path, target_size = TARGET_SIZE, batch_size = BATCH_SIZE, class_mode = None, seed = seed)
  return generator

def preprocess_segm(path):
  seed = 1
  datagen = ImageDataGenerator(rescale=1/127, dtype = 'int', rotation_range = 5)
  generator = datagen.flow_from_directory(directory = path, target_size = TARGET_SIZE, batch_size = BATCH_SIZE, class_mode = None, seed = seed)
  return generator
