from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import glob
from matplotlib import pyplot as plt

#FOLDER_NAME = 'GAN-TOY'
FOLDER_NAME = 'GAN-TL'

folderlist = glob.glob('GAN-TL/*')
print(folderlist)

#print(filelist)

# load and prepare the image
def load_image(filename):
	img = load_img(filename, target_size=(32, 32))
	img = img_to_array(img)
	img = img.reshape(1, 32, 32, 3)
	img = img.astype('float32')
	img = img / 255.0
	return img


model = load_model('final_model.h5')

for foldername in folderlist:
  filelist = glob.glob(foldername + '/*')
  images_count = len(filelist)
  correct = 0
  print(filelist)

  for fname in filelist:
    image = load_image(fname)

    result = model.predict_classes(image)

    #print(result[0])
    if (result[0][0]) == 1:
      correct = correct + 1

  if (images_count > 0):
    print(images_count, correct / images_count)
  else:
    print(images_count, 'NO IMAGES')