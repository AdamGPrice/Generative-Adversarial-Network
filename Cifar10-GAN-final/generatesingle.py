from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import glob
import os

FILEPATH = '2021-01-03 21-31'

class_labels = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def generate_latent_points(latent_dim, n_samples, n_classes):
	# random latent noise
	random_digits = randn(latent_dim * n_samples)
	latent_points = random_digits.reshape(n_samples, latent_dim)
	# random labels
	labels = randint(0, n_classes, n_samples)
	return [latent_points, labels]

def save_images(images, label):
    _, ax = plt.subplots(5, 7, figsize=(14,10))
    for i in range(5 * 7):
        ax[i//7, i%7].imshow(images[i])
        ax[i//7, i%7].axis('off')

    filename = '/misc2/%s' % class_labels[label]
    plt.savefig(FILEPATH + filename)
    plt.close()


## make folder for custom images
if not os.path.exists(FILEPATH + '/misc2'):
	os.makedirs(FILEPATH + '/misc2')


model_list = glob.glob(FILEPATH + '/models/*/')
model_dir = model_list[len(model_list) - 1]
generator = load_model(model_dir + '/generator.h5')

# generate images
#for class_label in range(len(class_labels)):
#	latent_points, _ = generate_latent_points(100, 25, 10)
#	labels = asarray([class_label for x in range(25)])
#	images = generator.predict([latent_points, labels])
#
#	images = (images + 1) / 2.0
#	save_images(images, class_label)

class_label = 1
latent_points, _ = generate_latent_points(100, 100, 10)
labels = asarray([class_label for x in range(100)])
images = generator.predict([latent_points, labels])
images = (images + 1) / 2.0
save_images(images, class_label)

class_label = 7
latent_points, _ = generate_latent_points(100, 100, 10)
labels = asarray([class_label for x in range(100)])
images = generator.predict([latent_points, labels])
images = (images + 1) / 2.0
save_images(images, class_label)