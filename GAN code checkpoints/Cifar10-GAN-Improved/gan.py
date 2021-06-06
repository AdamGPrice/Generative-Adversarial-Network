from numpy import expand_dims, zeros, ones, vstack
from numpy.random import randn, randint, random, choice

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2DTranspose, Flatten, LeakyReLU
from tensorflow.keras.layers import Reshape, Conv2D, Dropout, BatchNormalization
from tensorflow.keras.initializers import RandomNormal

from matplotlib import pyplot as plt
import time

### GPU MEMORY GROWTH SETUP
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Num GPUs Available:', len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

### LOAD THE DATASET
(train_images, _), (_, _) = cifar10.load_data()
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalise the data to [-1 , 1]

init = RandomNormal(mean=0.0, stddev=0.02)

### DEFINGE THE MODELS
def build_generator_model():
	model = Sequential()
	
	# foundation for 4x4 image
	model.add(Dense(512 * 4 * 4, kernel_initializer=init, input_dim=LATENT_DIM))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 512)))
	# upsample to 8x8
	model.add(Conv2DTranspose(256, (4,4), kernel_initializer=init, strides=(2,2), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 16x16
	model.add(Conv2DTranspose(128, (4,4), kernel_initializer=init, strides=(2,2), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), kernel_initializer=init, strides=(2,2), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# output layer
	model.add(Conv2D(3, (3,3), kernel_initializer=init, activation='tanh', padding='same'))
	return model

def build_discriminator_model():
	model = Sequential()
	# Input Layer
	model.add(Conv2D(64, (3,3), kernel_initializer=init, padding='same', input_shape=(32, 32, 3)))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(128, (3,3), kernel_initializer=init, strides=(2,2), padding='same'))
	#model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(128, (3,3), kernel_initializer=init, strides=(2,2), padding='same'))
	#model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(256, (3,3), kernel_initializer=init, strides=(2,2), padding='same'))
	#model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# Combine the generator and discriminator model to make a 'virtual' network 
def build_gan_model(generator, discriminator):
    discriminator.trainable = False
    
    ganModel = Sequential()
    ganModel.add(generator)
    ganModel.add(discriminator)

    optimizer = Adam(lr=0.0002, beta_1=0.5)
    ganModel.compile(loss="binary_crossentropy", optimizer=optimizer)
    return ganModel

def generate_latent_points(latent_dim, sample_count):
	random_nums = randn(latent_dim * sample_count)
	# reshape random numbers into inputs for the network
	latent_points = random_nums.reshape(sample_count, latent_dim)
	return latent_points

def smooth_positive_labels(y):
	return y - 0.3 + (random(y.shape) * 0.5)

def smooth_negative_labels(y):
	return y + random(y.shape) * 0.3

def noisy_labels(y, p_flip):
	flip_count = int(p_flip * y.shape[0])
	flip_indices = choice([i for i in range(y.shape[0])], size=flip_count)
	y[flip_indices] = 1 - y[flip_indices]
	return y

def get_real_samples(sample_count):
	i = randint(0, train_images.shape[0], sample_count)
	images = train_images[i]
	labels = ones((sample_count, 1))
	labels = smooth_positive_labels(labels)
	labels = noisy_labels(labels, 0.05)
	return images, labels

def get_fake_samples(generator, sample_count):
	latent_points = generate_latent_points(LATENT_DIM, sample_count)
	images = generator.predict(latent_points)
	labels = zeros((sample_count, 1))
	labels = smooth_negative_labels(labels)
	labels = noisy_labels(labels, 0.05)
	return images, labels

def save_generator_model(epoch, generator):
	filename = 'generator_model_%03d.h5' % (epoch+1)
	generator.save(filename)

def save_images(generator, epoch, n=10):
    images = generator.predict(SEED)	# Get the latent samples from our fixed seed

    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i])
    # save plot to file
    filename = 'images_at_epoch_%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.close()

def evalutate_performance(epoch, generator, discriminator, sample_count=150):
	# evaluate discriminator on real examples
	real_images, y_real = get_real_samples(sample_count)
	_, acc_real = discriminator.evaluate(real_images, y_real, verbose=0)

	# evaluate discriminator on fake examples
	fake_images, y_fake = get_fake_samples(generator, sample_count)
	_, acc_fake = discriminator.evaluate(fake_images, y_fake, verbose=0)

	# summarize discriminator performance
	print('Discriminator Accuracy - real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
	save_generator_model(epoch, generator)

def train(generator, discriminator, gan, epochs, batch_size):
	batches_per_epoch = int(train_images.shape[0] / batch_size)

	for epoch in range(epochs):
		start = time.time()
		for batch in range(batches_per_epoch):
			# Update the discriminator with real samples
			real_images, y_real = get_real_samples(int(batch_size/2))
			disc_loss1, _ = discriminator.train_on_batch(real_images, y_real)

			# Update the discriminator with fake samples
			fake_images, y_fake = get_fake_samples(generator, int(batch_size/2))
			disc_loss2, _ = discriminator.train_on_batch(fake_images, y_fake)

			# Update the generator using the gan model
			x_gan = generate_latent_points(LATENT_DIM, batch_size)
			y_gan = ones((batch_size, 1))
			generator_loss = gan.train_on_batch(x_gan, y_gan)

			print('Epoch %d/%d, %d/%d, Disc-Real=%.3f, Disc-Fake=%.3f Gen=%.3f' %
				(epoch+1, epochs, batch+1, batches_per_epoch, disc_loss1, disc_loss2, generator_loss))
		
		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
		# Save image every epoch and model every 5
		save_images(generator, epoch)
		if (epoch + 1) % 5 == 0:
			evalutate_performance(epoch, generator, discriminator)

### START OF PROGRAM
EPOCHS = 1000
BATCH_SIZE = 128
LATENT_DIM = 100
## Constant seed so we can see improvements
SEED = generate_latent_points(LATENT_DIM, BATCH_SIZE)



# Create the generator
generator = build_generator_model()
generator.summary()
# Create the discriminator
discriminator = build_discriminator_model()
discriminator.summary()
# Create the gan
gan = build_gan_model(generator, discriminator)
gan.summary()

# save_images(generator, -1, 10)

# Main function - begins training the gan
train(generator, discriminator, gan, EPOCHS, BATCH_SIZE)