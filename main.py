"""
# Commented out IPython magic to ensure Python compatibility.
! curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
! sudo apt-get install git-lfs
! git lfs install
! git clone https://github.com/neheller/kits19
# %cd kits19/
! python -m starter_code.get_imaging
"""

import nibabel as nib
import os
import matplotlib.pyplot as plt


def generate_volume(cid, base_path):
  #base_path wskazuje np na content/train
  data_path = Path('kits19/data')

  case_id = "case_{:05d}".format(cid)
  case_path = data_path / case_id
  vol = nib.load(str(case_path / "imaging.nii.gz"))

  vol = vol.get_fdata()

  check_dir(base_path)

  out_path = base_path / "vol"
  check_dir(out_path)

  for i in range(vol.shape[0]):
    fpath = out_path / (str(cid)+"_{:05d}.png".format(i))
    case_im = str(cid)+"_{:05d}.png".format(i)
    if case_im in os.listdir(out_path):
      print("picture already saved: ", case_im)
    else:
      plt.imsave(str(fpath), vol[i], cmap = 'gray')


def generate_segm(cid, base_path):
  #base_path wskazuje np na content/train
  data_path = Path('kits19/data')

  case_id = "case_{:05d}".format(cid)
  case_path = data_path / case_id
  segm = nib.load(str(case_path / "segmentation.nii.gz"))

  segm = segm.get_fdata()

  check_dir(base_path)

  out_path = base_path / "segm"
  check_dir(out_path)

  for i in range(segm.shape[0]):
    fpath = out_path / (str(cid)+"_{:05d}.png".format(i))
    case_im = str(cid)+"_{:05d}.png".format(i)
    if case_im in os.listdir(out_path):
      print("picture already saved: ", case_im)
    else:
      plt.imsave(str(fpath), segm[i], cmap = 'gray')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/
#zbior treningowy
from pathlib import Path
from preprocessing4 import check_dir
train_vol_path=Path('/content/train_vol')
check_dir(train_vol_path)
train_segm_path=Path('/content/train_segm')
check_dir(train_segm_path)
#zbior walidacyjny
val_vol_path=Path('/content/val_vol')
check_dir(val_vol_path)
val_segm_path=Path('/content/val_segm')
check_dir(val_segm_path)
#zbior testowy
test_vol_path=Path('/content/test_vol')
check_dir(test_vol_path)
test_segm_path=Path('/content/test_segm')
check_dir(test_segm_path)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/
#definiowanie zbiorow
TRAIN_CASES=30
VAL_CASES=15   #validation
TEST_CASES=10

for i in range(TRAIN_CASES):
  generate_volume(i,train_vol_path)
  generate_segm(i,train_segm_path)


for i in range(VAL_CASES):
  generate_volume(TRAIN_CASES+i, val_vol_path)
  generate_segm(TRAIN_CASES+i,val_segm_path)


for i in range(TEST_CASES):
  generate_volume(TRAIN_CASES+VAL_CASES+i, test_vol_path)
  generate_segm(TRAIN_CASES+VAL_CASES+i, test_segm_path)

#preprocessing zbiorów 
from preprocessing4 import preprocess_vol, preprocess_segm
train_X=preprocess_vol(train_vol_path)
train_Y=preprocess_segm(train_segm_path)

val_X=preprocess_vol(val_vol_path)
val_Y=preprocess_segm(val_segm_path)

test_X=preprocess_vol(test_vol_path)
test_Y=preprocess_segm(test_segm_path)

nb_train_samples=len(train_X.filenames)
nb_validation_samples=len(val_X.filenames)
nb_test_samples=len(test_X.filenames)

from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
from unet import *
import tensorflow as tf

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_WIDTH=128
IMG_HEIGHT=128
IMG_CHANNELS=3


def dice_coef(y_true, y_pred):
    smooth=1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels=3):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice

def multilabel_dice_loss(y_true, y_pred):
    return 1-dice_coef_multilabel(y_true, y_pred)

# combine generators into one which yields image and masks
train_set = zip(train_X,train_Y)
val_set = zip(val_X, val_Y)


inputs = tf.keras.layers.Input((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))
outputs = build_unet(inputs)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

model.compile(optimizer= 'adam', loss=multilabel_dice_loss,
              metrics=[dice_coef_multilabel])
model.summary()


batch_size=128
epochs=5

model.fit_generator(
    train_set,
    steps_per_epoch=nb_train_samples / batch_size,
    epochs=epochs,
    validation_data=val_set,
    validation_steps=nb_validation_samples / batch_size,
    callbacks=callbacks)

model.save('Model_save')

plt.figure()
plt.plot(model.history.history['loss'])
plt.ylabel('loss')
plt.xlabel('Epochs')
plt.show()

y_pred=model.predict_generator(test_X)