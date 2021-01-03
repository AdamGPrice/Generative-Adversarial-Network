from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import glob
import os

FILEPATH = '2021-01-03 00-04'

def generate_latent_points(latent_dim, n_samples, n_classes):
	# random latent noise
	random_digits = randn(latent_dim * n_samples)
	latent_points = random_digits.reshape(n_samples, latent_dim)
	# random labels
	labels = randint(0, n_classes, n_samples)
	return [latent_points, labels]

def save_images(images, n=10):
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(images[i])

	filename = '/misc/image'
	plt.savefig(FILEPATH + filename)
	plt.close()


## make folder for custom images
if not os.path.exists(FILEPATH + '/misc'):
	os.makedirs(FILEPATH + '/misc')


model_list = glob.glob(FILEPATH + '/models/*/')
model_dir = model_list[len(model_list) - 1]
generator = load_model(model_dir + '/generator.h5')

# generate images
latent_points, _ = generate_latent_points(100, 100, 10)
labels = asarray([7 for x in range(100)])
images = generator.predict([latent_points, labels])

images = (images + 1) / 2.0
save_images(images)