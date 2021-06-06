from tensorflow.keras.models import load_model
import numpy as np
import glob
from matplotlib import pyplot as plt

filelist = glob.glob('models/*')
print(filelist)