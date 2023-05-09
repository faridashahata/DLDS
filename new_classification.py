import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
#import tf.models as tfm
#import tensorflow_models as tfm
import pandas as pd
from sklearn.preprocessing import LabelEncoder


## FOLLOWING : https://www.kaggle.com/code/sanandachowdhury/transfer-learning-brain-tumor-classification


# data_list = []
# normal_list = []
# with open('data_list.txt', 'r') as f:
#     for line in f:
#         image_list = line.split(";")
#         if "_NORMAL" in image_list[1]:
#             normal_list.append([image_list[0], image_list[1].replace('\n', '')])
#         else:
#             data_list.append([image_list[0], image_list[1].replace('\n', '')])

# Read data into a dataframe:
data_df = pd.read_csv('data_list.txt', sep=";", header=None)
data_df.columns = ["image_name", "category"]
#print(data_df.head(20))

label_encoder = LabelEncoder()
data_df['label'] = label_encoder.fit_transform(data_df.category)
#print(data_df.head(5))

num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_

print("Num of classes", num_classes)
print("Classes:", class_names)

print(data_df['category'].value_counts().sort_values())



# Apparently tf creates classes in an alphanumeric fashion.
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='training_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    #batch_size=len(train),
    image_size=(224, 224))


validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory='validation_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    #batch_size=len(val),
    image_size=(224, 224))



val_batches = tf.data.experimental.cardinality(validation_ds)
print('Number of val batches: %d' % val_batches)
test_dataset = validation_ds.take(val_batches // 5)
validation_data = validation_ds.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_data))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


# NORMALIZE DATA:

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
val_ds = validation_data.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
test_ds =  test_dataset.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.


# Get EfficientNet V2 B0 here
efficientnet_v2_url = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2'
model_name = 'efficientnet_v2_b0'





# Set trainable to False for inference-only
set_trainable=False
import tensorflow_hub as hub
efficientnet_v2_b0 = hub.KerasLayer(efficientnet_v2_url, trainable=set_trainable, name=model_name)


def efficientnet_v2_model():
    #initializer = tf.keras.initializers.GlorotNormal()

    efficientnet_v2_sequential = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3), dtype=tf.float32, name='input_image'),
        efficientnet_v2_b0,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(44, dtype=tf.float32, activation='softmax')
    ], name='efficientnet_v2_sequential_model')

    return efficientnet_v2_sequential


# Generate Model
model_efficientnet_v2 = efficientnet_v2_model()

# Generate Summary of the Model
model_efficientnet_v2.summary()

print("len of validation ds", len(val_ds))


# Compile the model
model_efficientnet_v2.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics='acc'
)

history = model_efficientnet_v2.fit(train_ds,
                epochs=20,
                validation_data=val_ds,
                validation_steps=int(len(val_ds)),
                shuffle=False)


# After 10 epochs:  loss: 0.3545 - acc: 0.8822 - val_loss: 0.7879 - val_acc: 0.7911

print(model_efficientnet_v2.evaluate(test_ds))


# Epoch 20/20
# 112/112 [==============================]
# - 35s 312ms/step - loss: 0.1299 - acc: 0.9542 - val_loss: 0.6327 - val_acc: 0.8146

# After 20 epochs on test:
# 2/2 [==============================] - 1s 290ms/step - loss: 0.6322 - acc: 0.7812
# test loss and acc: [0.6321812272071838, 0.78125]