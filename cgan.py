# tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2DTranspose, Flatten, LeakyReLU
from tensorflow.keras.layers import Reshape, Conv2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Embedding, Input, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import RandomNormal

# utilities
from numpy import expand_dims, zeros, ones, vstack
from numpy.random import randn, randint, random, choice
from matplotlib import pyplot as plt
import time
import datetime
import os
import glob

EPOCHS = 1000
BATCH_SIZE = 128
LATENT_DIM = 100
CLASS_COUNT = 10

SEEDS = None
FILEPATH = None

### LOAD THE DATASET
(TRAIN_IMAGES, TRAIN_LABELS), (_, _) = cifar10.load_data()
TRAIN_IMAGES = TRAIN_IMAGES.astype('float32')
TRAIN_IMAGES = (TRAIN_IMAGES - 127.5) / 127.5 # Normalise the data to [-1 , 1]

### PROGRAM ENTRY POINT
def main():
	# generate seeds
	global SEEDS
	SEEDS = generate_latent_points(LATENT_DIM, 100, CLASS_COUNT)

	# generate or load the models
	generator, discriminator, gan = model_setup()
	save_generator_images(generator, 0)
	
	# begin the training process
	train(generator, discriminator, gan, EPOCHS, BATCH_SIZE)


### DEFINE THE MODELS
init = RandomNormal(mean=0.0, stddev=0.02)

def build_generator_model(n_classes):
	## label input
	in_label = Input(shape=(1,))
	label_layers = Embedding(n_classes, 50)(in_label)
	label_layers = Dense(4 * 4 * 3, kernel_initializer=init)(label_layers)
	# reshape label
	label_layers = Reshape((4, 4, 3))(label_layers)

	## latent input
	in_latent = Input(shape=(100,))
	# foundation for 4x4 image
	latent_layers = Dense(512 * 4 * 4, kernel_initializer=init)(in_latent)
	latent_layers = LeakyReLU(alpha=0.2)(latent_layers)
	latent_layers = Reshape((4, 4, 512))(latent_layers)

	## merge input and latent points
	gen = Concatenate()([latent_layers, label_layers])
	# upsample to 8x8
	gen = Conv2DTranspose(256, (4,4), strides=(2,2), kernel_initializer=init, padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 16x16
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), kernel_initializer=init, padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 32x32
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), kernel_initializer=init, padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(3, (3,3), activation='tanh', kernel_initializer=init, padding='same')(gen)

	# define the model
	model = Model([in_latent, in_label], out_layer)
	return model

def build_discriminator_model(n_classes):
	## label input
	in_label = Input(shape=(1,))
	label_layers = Embedding(n_classes, 50)(in_label)
	label_layers = Dense(32 * 32 * 3, kernel_initializer=init)(label_layers)
	label_layers = Reshape((32, 32, 3))(label_layers)

	## image input
	in_image = Input(shape=(32, 32, 3))
	
	## merge label and image layers
	disc = Concatenate()([in_image, label_layers])
	# downsample
	disc = Conv2D(64, (3,3), strides=(2,2), kernel_initializer=init, padding='same')(disc)
	disc = LeakyReLU(alpha=0.2)(disc)
	# downsample
	disc = Conv2D(128, (3,3), strides=(2,2), kernel_initializer=init, padding='same')(disc)
	disc = LeakyReLU(alpha=0.2)(disc)
	# downsample
	disc = Conv2D(256, (3,3), strides=(2,2), kernel_initializer=init, padding='same')(disc)
	disc = LeakyReLU(alpha=0.2)(disc)
	# downsample
	disc = Conv2D(512, (3,3), strides=(2,2), kernel_initializer=init, padding='same')(disc)
	disc = LeakyReLU(alpha=0.2)(disc)

	# output
	disc = Flatten()(disc)
	disc = Dropout(0.4)(disc)
	out_layer = Dense(1, kernel_initializer=init, activation='sigmoid')(disc)

	# define the model
	model = Model([in_image, in_label], out_layer)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def build_gan_model(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False

	# model the gan as the generator and discriminator connected
	gen_latent, gen_label = generator.input
	gen_output = generator.output
	gan_output = discriminator([gen_output, gen_label])
	model = Model([gen_latent, gen_label], gan_output)
	
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


### TRAINING FUNCTIONS
def generate_latent_points(latent_dim, n_samples, n_classes):
	# random latent noise
	random_digits = randn(latent_dim * n_samples)
	latent_points = random_digits.reshape(n_samples, latent_dim)
	# random labels
	labels = randint(0, n_classes, n_samples)
	return [latent_points, labels]

def smooth_positive_targets(targets):
	return targets - 0.3 + (random(targets.shape) * 0.5)

def smooth_negative_targets(targets):
	return targets + random(targets.shape) * 0.3

def flip_targets(targets, flip_percentage):
	count = int(flip_percentage * targets.shape[0])
	indices = choice([i for i in range(targets.shape[0])], size=count)
	targets[indices] = 1 - targets[indices]
	return targets

def get_real_samples(n_samples, noise=True):
	indices = randint(0, TRAIN_IMAGES.shape[0], n_samples)
	images, labels = TRAIN_IMAGES[indices], TRAIN_LABELS[indices]
	targets = ones((n_samples, 1))
	if noise == True:
		targets = smooth_positive_targets(targets)
		targets = flip_targets(targets, 0.04)
	return [images, labels], targets

def get_fake_samples(generator, n_samples, noise=True):
	latent_points, labels = generate_latent_points(LATENT_DIM, n_samples, CLASS_COUNT)
	images = generator.predict([latent_points, labels])
	targets = zeros((n_samples, 1))
	if noise == True:
		targets = smooth_negative_targets(targets)
		targets = flip_targets(targets, 0.04)
	return [images, labels], targets

def evaluate_performance(generator, discriminator, epoch, n_samples=100):
	# evaluate discriminator on real samples
	[r_images, r_labels], r_targets = get_real_samples(n_samples, False)
	_, real_acc = discriminator.evaluate([r_images, r_labels], r_targets)

	# evaluate discriminator on fake samples
	[f_images, f_labels], f_targets = get_fake_samples(generator, n_samples, False)
	_, fake_acc = discriminator.evaluate([f_images, f_labels], f_targets)

	# summarize performance
	print('Disc Accuracy - real: %.0f%%, fake: %.0f%%' % (real_acc * 100, fake_acc * 100))
	save_models(generator, discriminator, epoch) # could probs just have this in the main loop

def train(generator, discriminator, gan, epochs, batch_size):
	batches_per_epoch = int(TRAIN_IMAGES.shape[0] / batch_size)

	for epoch in range(epochs):
		start = time.time()
		for batch in range(batches_per_epoch):
			# update the discriminator with real samples
			[r_images, r_labels], r_targets = get_real_samples(int(batch_size/2))
			d_loss1, _ = discriminator.train_on_batch([r_images, r_labels], r_targets)

			# update the discriminator with fake samples
			[f_images, f_labels], f_targets = get_fake_samples(generator, int(batch_size/2))
			d_loss2, _ = discriminator.train_on_batch([f_images, f_labels], f_targets)

			# update the generator using the gan model
			latent_points, labels = generate_latent_points(LATENT_DIM, batch_size, CLASS_COUNT)
			targets = ones((batch_size, 1))
			g_loss = gan.train_on_batch([latent_points, labels], targets)

			print('Epoch %d/%d, %d/%d, Disc-Real=%.3f, Disc-Fake=%.3f Gen=%.3f' %
				(epoch+1, epochs, batch+1, batches_per_epoch, d_loss1, d_loss2, g_loss))

		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

		save_generator_images(generator, epoch)
		if (epoch + 1) % 1 == 0:
			evaluate_performance(generator, discriminator, epoch)

### LOAD AND SAVE FUNCTIONS
def model_setup():
	# create a new filepath if none is set or load models inside the specified path
	global FILEPATH
	if FILEPATH == None:
		now = datetime.datetime.now()
		FILEPATH = '{} {:02d}-{:02d}'.format(now.date(), now.hour, now.minute)
		if not os.path.exists(FILEPATH):
			os.makedirs(FILEPATH)
			os.makedirs(FILEPATH + '/' + 'images')
			os.makedirs(FILEPATH + '/' + 'models')

		# generate the models
		generator = build_generator_model(CLASS_COUNT)
		generator.summary()
		discriminator = build_discriminator_model(CLASS_COUNT)
		discriminator.summary()
		gan = build_gan_model(generator, discriminator)
		gan.summary()

		return generator, discriminator, gan
	else:
		print('Loading pre-trained models')
		model_list = glob.glob(FILEPATH + '/models/*/')
		model_dir = model_list[len(model_list) - 1]
		generator = load_model(model_dir + '/generator.h5')
		generator.summary()
		discriminator = load_model(model_dir + '/discriminator.h5')
		discriminator.summary()

		opt = Adam(lr=0.0002, beta_1=0.5)
		discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

		gan = build_gan_model(generator, discriminator)
		gan.summary()

		return generator, discriminator, gan

def save_models(generator, discriminator, epoch):
	directory = FILEPATH + '/models/epoch_%04d' % (epoch+1)
	os.makedirs(directory)
	g_dir = directory + '/generator.h5'
	generator.save(g_dir)
	d_dir = directory + '/discriminator.h5'
	discriminator.save(d_dir)


### IMAGE FUNCTIONS
def save_generator_images(generator, epoch, n=10):
	if SEEDS == None:
		latent_points, labels = generate_latent_points(LATENT_DIM, 100, CLASS_COUNT)
		images = generator.predict([latent_points, labels])
	else:
		images = generator.predict(SEEDS)

	# scale from -1, 1 to 0, 1
	images = (images + 1) / 2.0
	# plot images
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(images[i])

	# save plot to file
	filename = 'images/epoch_%04d.png' % (epoch+1)
	plt.savefig(FILEPATH + '/' + filename)
	plt.close()



main()