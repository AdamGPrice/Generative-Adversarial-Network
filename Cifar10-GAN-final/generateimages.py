## For generating images at every epoch

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import save_img
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
from PIL import Image


FILEPATH = '2021-01-03 21-31'

def generate_latent_points(latent_dim, n_samples, n_classes):
	# random latent noise
	random_digits = randn(latent_dim * n_samples)
	latent_points = random_digits.reshape(n_samples, latent_dim)
	# random labels
	labels = randint(0, n_classes, n_samples)
	return [latent_points, labels]

modellist = glob.glob(FILEPATH + '/models/*')
print(modellist)

## make folder for Batched images
if not os.path.exists(FILEPATH + '/benchmark-images'):
  os.makedirs(FILEPATH + '/benchmark-images')


for modelfolder in modellist:
  epoch = modelfolder.split('\\')[1]
  model = load_model(modelfolder + '/generator.h5')

  if not os.path.exists(FILEPATH + '/benchmark-images/' + epoch):
    os.makedirs(FILEPATH + '/benchmark-images/' + epoch)

  class_label = 7
  count = 500
  latent_points, _ = generate_latent_points(100, count, 10)
  labels = asarray([class_label for x in range(count)])
  images = model.predict([latent_points, labels])
  images = ((images + 1) / 2.0) * 255.0

  for i in range(len(images)):
    save_img(FILEPATH + '/benchmark-images/' + epoch + '/'+ str(i) + '.png', images[i])