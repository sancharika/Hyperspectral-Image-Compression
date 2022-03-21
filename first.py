import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import scipy.io as spio
import scipy.stats
import tensorflow
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Resizing
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from attention import attention
import math
from scipy.stats import zscore
#%%
def custom_loss(Y_true, Y_pred):
    PSNR = tf.image.psnr(Y_true, Y_pred, max_val = 255)
    return PSNR

#%%
data = spio.loadmat("Indian_pines.mat")["indian_pines"]   
data_gt = spio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
data_norm = zscore(data)
#data_norm = np.stack((data_norm, data_norm))
data_norm = np.reshape(data_norm, (1,145,145,220))

#%%
#from spectral import *
import tifffile
tifffile.imshow(data_norm[:,:,:,7])
#tifffile.imshow(data)
#tifffile.imshow(data_gen)


#%%
opt = Adam(lr=0.0009, beta_1=0.9,beta_2 = 0.999)
encoder = Sequential()

#encoder.add(Conv2D(220,( 3, 3),activation = 'relu'))  
#encoder.add(MaxPooling2D(pool_size = (2, 2))) 

encoder.add(Conv2D(128,( 3, 3),activation = 'relu'))  
encoder.add(MaxPooling2D(pool_size = (2, 2))) 
encoder.add(Conv2D(64,( 3, 3), activation = 'relu'))  
encoder.add(MaxPooling2D(pool_size = (2, 2)))  
#encoder.add(tf.keras.layers.Reshape((34,34,64)))


encoder.add(Conv2D(32,( 3, 3), activation = 'relu'))  
encoder.add(MaxPooling2D(pool_size = (2, 2)))
encoder.add(attention())

encoder.add(Flatten())
encoder.add(Dense(units = 8192,activation = 'relu'))   

encoder.compile(optimizer=opt)

decoder = Sequential()
#decoder.add(tf.keras.layers.Reshape((16,16,32)))
#decoder.add(attention())

decoder.add(Dense(units = 8192, activation = 'relu'))

decoder.add(tf.keras.layers.Reshape((16,16,32)))
decoder.add(UpSampling2D(size = (2,2)))
decoder.add(Conv2DTranspose(64,(3, 3), activation = "relu"))    
decoder.add(UpSampling2D(size = (2,2)))
decoder.add(Conv2DTranspose(64,(3, 3), activation = "relu"))
decoder.add(UpSampling2D(size = (2,2)))

decoder.add(Conv2DTranspose(128,(3, 3), activation = "relu"))
#decoder.add(UpSampling2D(size = (2,2)))

decoder.add(Conv2DTranspose(220,(3,3), activation = "relu"))
decoder.add(Conv2DTranspose(220,(3,3), activation = "linear"))
decoder.add(Resizing(145,145))
decoder.compile(optimizer=opt)
    
model = Sequential()
model.add(encoder)
model.add(decoder)

model.compile(optimizer = opt, metrics = "mse", loss = "mse")

model.fit(data_norm, data_norm, epochs = 350)
    
data_gen = model.predict(data_norm)

print(custom_loss(data_norm, data_gen))

#model.save("IP_try2.h5")
#encoder.save("IP_try2_enc.h5")
#decoder.save("IP_try2_dec.h5")
data_enc = encoder.predict(data_norm)
tifffile.imshow(data_gen)

