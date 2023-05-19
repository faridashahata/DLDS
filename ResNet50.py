from matplotlib import pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2

# import tf.models as tfm
# import tensorflow_models as tfm
# import tensorflow_hub as hub
from typing import *
from tqdm import tqdm
import shutil

data_list = []
normal_list = []
with open("data_list.txt", "r") as f:
    for line in f:
        image_list = line.split(";")
        if "_NORMAL" in image_list[1]:
            normal_list.append([image_list[0], image_list[1].replace("\n", "")])
        else:
            data_list.append([image_list[0], image_list[1].replace("\n", "")])


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

    val_list_unnormal = data_list[
        train_lim_unnormal : train_lim_unnormal + val_lim_unnormal
    ]
    val_list_normal = normal_list[train_lim_normal : train_lim_normal + val_lim_normal]
    val_list = [*val_list_unnormal, *val_list_normal]

    test_list_unnormal = data_list[train_lim_unnormal + val_lim_unnormal :]
    test_list_normal = normal_list[train_lim_normal + val_lim_normal :]
    test_list = [*test_list_unnormal, *test_list_normal]

    return train_list, val_list, test_list


train, val, test = store_train_val(data_list, normal_list, 0.8, 0.2, 0.0)


print("len of train", len(train))
print("len of val", len(val))
print("len of test", len(test))


# CREATE FIRST FOLDERS: training_data and validation_data:


def create_files(file_names, folder_path):
    path_extract = "resized_images/"
    filenames = os.listdir(path_extract)
    # new_path = 'training_data/'
    new_path = folder_path

    # instead of file_names, put train, val or test:
    for image in file_names:
        image_name = image[0]
        image_class = image[1]

        im_path = os.path.join(path_extract, image_class, image_name)
        new_folder_path = os.path.join(new_path, image_class)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        new_im_path = os.path.join(new_folder_path, image_name)
        image_file = cv2.imread(im_path)
        cv2.imwrite(new_im_path, image_file)

    # Make sure the folder contains a file for each of the 44 categories:
    for file in filenames:
        new_folder_path = os.path.join(new_path, file)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

    return


# Run this on a newly created training data folder:


def delete_folders_with_few_images(path_to_training: str, path_to_valid: str):
    categories_to_remove = []
    if not os.path.exists(path_to_training):
        print("Path does not exist")

    folders: List[str] = os.listdir(path_to_training)

    for folder_name in tqdm(folders):
        folder_name: str

        folder_path: str = os.path.join(path_to_training, folder_name)

        if os.path.isdir(folder_path):
            files: List[str] = os.listdir(folder_path)
            # print("len files", len(files))
            if len(files) < 20:
                # os.rmdir(folder_path)

                shutil.rmtree(folder_path, ignore_errors=True)
                categories_to_remove.append(folder_name)

    folders_in_val = os.listdir(path_to_valid)
    for folder_name in tqdm(folders):
        if folder_name in categories_to_remove:
            folder_path = os.path.join(path_to_valid, folder_name)
            # os.rmdir(folder_path)
            shutil.rmtree(folder_path, ignore_errors=True)
    print("categories to remove", categories_to_remove)
    return


# categories_to_remove = delete_folders_with_few_images('training_data_2/','validation_data_2/' )


# # Run the following lines once (alternatively clear files for different runs/draws of data splits):
# create_files(train, 'training_data/')
# create_files(val, 'validation_data/')
#

# This is to check if a file is empty or not incase we want to either repopulate it or delete its contents for new runs
# print(os.path.getsize(new_path) == 0)
# print(len(os.listdir(new_path)))


# STEP 1: Read data from directory:
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="training_data/",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(224, 224),
)


validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory="validation_data/",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(224, 224),
)


class_names = np.array(train_ds.class_names)
print("class names", class_names)


# STEP 2: Create test data:
val_batches = tf.data.experimental.cardinality(validation_ds)
print("Number of val batches: %d" % val_batches)
test_dataset = validation_ds.take(val_batches // 5)
validation_data = validation_ds.skip(val_batches // 5)

print(
    "Number of validation batches: %d"
    % tf.data.experimental.cardinality(validation_data)
)
print("Number of test batches: %d" % tf.data.experimental.cardinality(test_dataset))

# Build Augmentation layer:

augmentation_layer = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomZoom(0.1),
        # tf.keras.layers.RandomCrop(0.1),
    ],
    name="augmentation_layer",
)


# STEP 3: Normalize data:
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(
    lambda x, y: (tf.keras.applications.resnet.preprocess_input(x), y)
)  # Where x—images, y—labels.
val_ds = validation_data.map(
    lambda x, y: (tf.keras.applications.resnet.preprocess_input(x), y)
)  # Where x—images, y—labels.
test_ds = test_dataset.map(
    lambda x, y: (tf.keras.applications.resnet.preprocess_input(x), y)
)  # Where x—images, y—labels.
val_ds = validation_data
test_ds = test_dataset

AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# STEP 4: Get pre-trained models from this link:
mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
resnet50 = "https://tfhub.dev/tensorflow/resnet_50/classification/1"
resnet50_v2 = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5"
IMAGE_SHAPE = (224, 224)

feature_extractor_model = resnet50
# ResNetRS152
base_model = tf.keras.applications.ResNetRS50(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)

base_model.trainable = True

#
# feature_extractor_layer = hub.KerasLayer(
#     feature_extractor_model,
#     #input_shape=(224, 224, 3),
#     trainable=False)


base_model.summary()

# Fine-tune from this layer onwards
fine_tune_at = 700
# fine_tune_at = 781-10

# 771,776
# fine_tune_at = 70

print("base model layers", len(base_model.layers))
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


# STEP 5: Build Model:
num_classes = len(class_names)
# initializer = tf.keras.initializers.GlorotNormal(seed=31)

# preprocess_input = tf.keras.applications.resnet50.preprocess_input
# x = preprocess_input(train_ds)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(224, 224, 3), dtype=tf.float32, name="input_image"
        ),
        # feature_extractor_layer,
        augmentation_layer,
        base_model,
        global_average_layer,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(
            512, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.1)
        ),
        tf.keras.layers.Dense(
            256, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.05)
        ),
        tf.keras.layers.Dense(num_classes, dtype=tf.float32, activation="softmax"),
    ]
)

model.summary()

# STEP 6: Compile the model:
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.8
)
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # loss='categorical_crossentropy',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

NUM_EPOCHS = 20

# STEP 7: Fit the model:
history = model.fit(
    train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=[early_stoppage]
)

# STEP 8: Evaluate:
print(model.evaluate(test_ds))

# lr=0.01
# Epoch 20/20
# 112/112 [==============================] - 130s 1s/step - loss: 0.7740 - acc: 0.7400 - val_loss: 1.7613 - val_acc: 0.6005
# Test loss and acc: [2.276601552963257, 0.578125]

# 40 epochs: lr=0.01
# Epoch 40/40
# 112/112 [==============================] - 137s 1s/step - loss: 0.4745 - acc: 0.8419 - val_loss: 2.1517 - val_acc: 0.6188
# Test: [1.7676501274108887, 0.59375]


# 5 epochs, ResnetRs101:
# Epoch 5/5
# 112/112 [==============================] - 1047s 9s/step - loss: 0.2722 - acc: 0.9154 - val_loss: 0.5575 - val_acc: 0.8599
# #[0.48008185625076294, 0.862500011920929]


# 5 epochs, ResnetRS152:
# Epoch 5/5
# 112/112 [==============================] - 2965s 27s/step - loss: 0.2320 - acc: 0.9232 - val_loss: 0.3929 - val_acc: 0.8871
# [0.2928621470928192, 0.9312499761581421]


# 10 epochs, ResnetRS152:
# Epoch 10/10
# 112/112 [==============================] - 1560s 14s/step - loss: 0.0847 - acc: 0.9715 - val_loss: 0.2813 - val_acc: 0.9197

# 5/5 [==============================] - 13s 2s/step - loss: 0.3066 - acc: 0.9312
# [0.3065941631793976, 0.9312499761581421]

# STEP 9: Plot loss and accuracies:

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()


# USE:  https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub

# useful tutorial, another model: https://www.kaggle.com/code/bulentsiyah/dogs-vs-cats-classification-vgg16-fine-tuning

# Tutorial using pretrained convnext
# https://www.kaggle.com/code/shrijeethsuresh/brain-tumor-convnext-44-classes-99-4-acc

# Tutorial with transfer learning on tumor data: To follow general steps in:
# https://www.kaggle.com/code/sanandachowdhury/transfer-learning-brain-tumor-classification


# FINAL EXPERIMENT:
# ResnetRS152, 10 epochs,  learning_rate=0.001, fine_tune_at = 700, no scheduler

# 5/5 [==============================] - 13s 3s/step - loss: 0.3033 - acc: 0.9438
# [0.3032929301261902, 0.9437500238418579]
#
# Epoch 10/10
# 112/112 [==============================] - 472s 4s/step - loss: 0.0491 - acc: 0.9846 - val_loss: 0.2762 - val_acc: 0.9279
