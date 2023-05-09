import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
import pandas as pd
import tensorflow_hub as hub

# STEP 1: Read data from directory:

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='training_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224))


validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory='validation_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224))


# STEP 2: Create test data:
val_batches = tf.data.experimental.cardinality(validation_ds)
print('Number of val batches: %d' % val_batches)
test_dataset = validation_ds.take(val_batches // 5)
validation_data = validation_ds.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_data))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


# STEP 3: Normalize data:

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
val_ds = validation_data.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
test_ds = test_dataset.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.


# STEP 4: Get EfficientNet V2 from this link:
model_link = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2'
model_layer = hub.KerasLayer(model_link, trainable=False)

# STEP 5: Generate Model:
model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3), dtype=tf.float32, name='input_image'),
        model_layer,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(44, dtype=tf.float32, activation='softmax')
    ])

model.summary()

# STEP 6: Compile the model:
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics='acc'
)
# STEP 7: Fit the model:
history = model.fit(train_ds,
                epochs=30,
                validation_data=val_ds,
                validation_steps=int(len(val_ds)),
                shuffle=False)


# STEP 8: Evaluate:
# lr=0.001
# After 10 epochs:  loss: 0.3545 - acc: 0.8822 - val_loss: 0.7879 - val_acc: 0.7911

print(model.evaluate(test_ds))


# Epoch 20/20 lr=0.001
# 112/112 [==============================]
# - 35s 312ms/step - loss: 0.1299 - acc: 0.9542 - val_loss: 0.6327 - val_acc: 0.8146

# After 20 epochs on test: lr=0.001
# 2/2 [==============================] - 1s 290ms/step - loss: 0.6322 - acc: 0.7812
# [0.6321812272071838, 0.78125]

#------------------------------------------------------------------------------------
# After 30 epochs on test: lr=0.001:
# Epoch 30/30
# 112/112 [==============================] - 36s 325ms/step - loss: 0.1117 - acc: 0.9615 - val_loss: 0.6393 - val_acc: 0.8433
# 2/2 [==============================] - 1s 291ms/step - loss: 0.2503 - acc: 0.9062
# [0.2502741813659668, 0.90625]



# STEP 9: Plot loss and accuracies:
from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()