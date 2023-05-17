import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
#import tf.models as tfm
#import tensorflow_models as tfm
import tensorflow_hub as hub
from typing import *
from tqdm import tqdm
import shutil

data_list = []
normal_list = []
with open('data_list.txt', 'r') as f:
    for line in f:
        image_list = line.split(";")
        if "_NORMAL" in image_list[1]:
            normal_list.append(
                [image_list[0], image_list[1].replace('\n', '')])
        else:
            data_list.append([image_list[0], image_list[1].replace('\n', '')])


def store_train_val(data_list, normal_list, train_prop, val_prop, test_prop):
    if (train_prop + val_prop + test_prop) != 1:
        raise ("The sum of the proportions must be 1")

    train_list = []
    val_list = []
    test_list = []

    np.random.shuffle(data_list)
    np.random.shuffle(normal_list)

    n = len(data_list)
    m = len(normal_list)

    train_lim_unnormal = int(train_prop * n)
    train_lim_normal = int(train_prop * m)
    val_lim_unnormal = int(val_prop * n)
    val_lim_normal = int(val_prop * m)

    train_list_unnormal = data_list[:train_lim_unnormal]
    train_list_normal = normal_list[:train_lim_normal]
    train_list = [*train_list_unnormal, *train_list_normal]

    val_list_unnormal = data_list[train_lim_unnormal:
                                  train_lim_unnormal + val_lim_unnormal]
    val_list_normal = normal_list[train_lim_normal:
                                  train_lim_normal + val_lim_normal]
    val_list = [*val_list_unnormal, *val_list_normal]

    test_list_unnormal = data_list[train_lim_unnormal + val_lim_unnormal:]
    test_list_normal = normal_list[train_lim_normal + val_lim_normal:]
    test_list = [*test_list_unnormal, *test_list_normal]

    return train_list, val_list, test_list


train, val, test = store_train_val(data_list, normal_list, 0.8, 0.2, 0.0)

#
# print("len of train", len(train))
# print("len of val", len(val))
# print("len of test", len(test))


# CREATE FIRST FOLDERS: training_data_binary and val_data_binary:

def create_files_binary(file_names, folder_path):
    path_extract = "resized_images/"
    filenames = os.listdir(path_extract)
    #new_path = 'training_data/'
    new_path = folder_path

    # instead of file_names, put train, val or test:
    for image in file_names:
        image_name = image[0]
        image_class = image[1]
        im_path = os.path.join(path_extract, image_class, image_name)
        if "_NORMAL" in image_class:
            print("image class", image_class)
            bin_image_class = "NORMAL"
            new_folder_path = os.path.join(new_path, bin_image_class)
        else:
            bin_image_class = "TUMOR"
            new_folder_path = os.path.join(new_path, bin_image_class)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        new_im_path = os.path.join(new_folder_path, image_name)
        image_file = cv2.imread(im_path)
        cv2.imwrite(new_im_path, image_file)

    # # Make sure the folder contains a file for each of the 44 categories:
    # for file in filenames:
    #     new_folder_path = os.path.join(new_path, file)
    #     if not os.path.exists(new_folder_path):
    #         os.makedirs(new_folder_path)

    return

# Run the following lines once (alternatively clear files for different runs/draws of data splits):
# create_files_binary(train, 'training_data_binary/')
# create_files_binary(val, 'val_data_binary/')



# STEP 1: Read data from directory:
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='training_data_binary/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224))


validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory='val_data_binary/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224))


class_names = np.array(train_ds.class_names)
print("class names", class_names)


# STEP 2: Create test data:
val_batches = tf.data.experimental.cardinality(validation_ds)
print('Number of val batches: %d' % val_batches)
test_dataset = validation_ds.take(val_batches // 5)
validation_data = validation_ds.skip(val_batches // 5)

print('Number of validation batches: %d' %
      tf.data.experimental.cardinality(validation_data))
print('Number of test batches: %d' %
      tf.data.experimental.cardinality(test_dataset))



# STEP 3: Normalize data:
normalization_layer = tf.keras.layers.Rescaling(1./255)

# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
# val_ds = validation_data.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
# test_ds = test_dataset.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.

# TRY USING PREPROCESS INPUTS:
train_ds = train_ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y)) # Where x—images, y—labels.
val_ds = validation_data.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y)) # Where x—images, y—labels.
test_ds = test_dataset.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y)) # Where x—images, y—labels.



# val_ds = validation_data
# test_ds = test_dataset



# STEP 4: Get pre-trained models from this link:
mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
resnet50 = "https://tfhub.dev/tensorflow/resnet_50/classification/1"
resnet50_v2 = 'https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5'
IMAGE_SHAPE = (224, 224)

#feature_extractor_model = resnet50
#ResNetRS152
base_model = tf.keras.applications.resnet50.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

#base_model.trainable = True

base_model.trainable = False

#
# feature_extractor_layer = hub.KerasLayer(
#     feature_extractor_model,
#     #input_shape=(224, 224, 3),
#     trainable=False)


# feature_extractor_model.summary()

# Fine-tune from this layer onwards
#fine_tune_at = 45
#fine_tune_at = 130
#fine_tune_at = 90

#fine_tune_at = 130

fine_tune_at = 165


print("base model layers", len(base_model.layers))
# Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


# STEP 5: Build Model:
num_classes = len(class_names)
#initializer = tf.keras.initializers.GlorotNormal(seed=31)

# preprocess_input = tf.keras.applications.resnet50.preprocess_input
#x = preprocess_input(train_ds)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3),
                          dtype=tf.float32, name='input_image'),
    # feature_extractor_layer,
    base_model,
    global_average_layer,
    #tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(
        num_classes, dtype=tf.float32, activation='softmax')
])

model.summary()

# STEP 6: Compile the model:
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
#optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

# Define Early Stopping Callback
earlystopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3)
callbacks_list = [earlystopping_callback]



model.compile(
    # 0.001
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #loss='binary_crossentropy',
    #loss=tf.keras.losses.CategoricalCrossentropy(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=['acc'])

NUM_EPOCHS = 10
CFG_SEED=71
tf.random.set_seed(CFG_SEED)


# STEP 7: Fit the model:
history = model.fit(train_ds,
                    validation_data=val_ds,
                   epochs=NUM_EPOCHS)
                    #, callbacks=callbacks_list)

# STEP 8: Evaluate:
print(model.evaluate(test_ds))


# STEP 9: Plot loss and accuracies:
from matplotlib import pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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




# 5 epochs, ResNetRS101, adam ,learning_rate=0.001,  fine_tune_at = 90:
#No

# FIRST EXPERIMENT:
# 2 epochs, resnet50 , lr_scheduler with initial lr : 1e-4, fine_tune_at = 45 (with dropout layer):
# Epoch 2/2
# 107/107 [==============================] - 344s 3s/step - loss: 0.0282 - acc: 0.9882 - val_loss: 0.1193 - val_acc: 0.9607

# 5/5 [==============================] - 5s 901ms/step - loss: 0.1313 - acc: 0.9500
# [0.13128714263439178, 0.949999988079071]


# SECOND EXPERIMENT: 98%
# 4 epochs, resnet50 , lr_scheduler with initial lr : 1e-4, fine_tune_at = 45 (without dropout layer):
# 107/107 [==============================] - 352s 3s/step - loss: 0.1256 - acc: 0.9484 - val_loss: 0.3344 - val_acc: 0.9173
# Epoch 2/4
# 107/107 [==============================] - 355s 3s/step - loss: 0.0267 - acc: 0.9941 - val_loss: 0.0493 - val_acc: 0.9818
# Epoch 3/4
# 107/107 [==============================] - 358s 3s/step - loss: 0.0145 - acc: 0.9962 - val_loss: 0.0272 - val_acc: 0.9902
# Epoch 4/4
# 107/107 [==============================] - 349s 3s/step - loss: 0.0124 - acc: 0.9965 - val_loss: 0.0370 - val_acc: 0.9888

# 5/5 [==============================] - 5s 921ms/step - loss: 0.0466 - acc: 0.9812
# [0.04655037075281143, 0.981249988079071]


