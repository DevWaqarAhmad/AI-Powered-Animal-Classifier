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

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,             # 80% training, 20% validation
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'                 # 80% data
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'              # 20% data
)