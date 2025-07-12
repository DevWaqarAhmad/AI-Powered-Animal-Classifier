import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

dataset_path = 'dataset'                    
model_save_path = 'model/animal_model.h5'

IMAGE_SIZE = (150, 150)   
BATCH_SIZE = 32           
EPOCHS = 10         