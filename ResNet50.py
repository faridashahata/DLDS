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


#base_model.trainable = True
base_model.trainable = False

#
# feature_extractor_layer = hub.KerasLayer(
#     feature_extractor_model,
#     #input_shape=(224, 224, 3),
#     trainable=False)


#base_model.summary()

# Fine-tune from this layer onwards
#fine_tune_at = 700
fine_tune_at = 1400
#fine_tune_at = 781-10

# 771,776
# fine_tune_at = 70

print("base model layers", len(base_model.layers))
# Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False
#
# Unfreeze BN layers to see the difference:
counter = 0
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        counter +=1
        layer.trainable = True
print(f"Make {counter} BN layers unfrozen")
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

#model.summary()

# STEP 6: Compile the model:
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.8
)
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

CFG_SEED=71
NUM_EPOCHS = 10
#tf.random.set_seed(CFG_SEED)

# Define Early Stopping Callback
earlystopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5)
callbacks_list = [earlystopping_callback]


# Try AdamW and lr sched
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),

    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # loss='categorical_crossentropy',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)



# STEP 7: Fit the model:
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=NUM_EPOCHS)
                    #,
                    #callbacks=callbacks_list)

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




# ResnetRS152, lr=0.01, no droupout or dense layer 512, 10 epochs, adam: with seed=71
# 112/112 [==============================] - 447s 4s/step - loss: 2.1962 - acc: 0.4273 - val_loss: 1.0751 - val_acc: 0.6844
# Epoch 2/10
# 112/112 [==============================] - 442s 4s/step - loss: 0.8122 - acc: 0.7476 - val_loss: 0.7620 - val_acc: 0.7551
# Epoch 3/10
# 112/112 [==============================] - 438s 4s/step - loss: 0.4584 - acc: 0.8565 - val_loss: 0.5187 - val_acc: 0.8449
# Epoch 4/10
# 112/112 [==============================] - 438s 4s/step - loss: 0.2672 - acc: 0.9143 - val_loss: 0.5414 - val_acc: 0.8571
# Epoch 5/10
# 112/112 [==============================] - 443s 4s/step - loss: 0.1519 - acc: 0.9514 - val_loss: 0.4029 - val_acc: 0.8952
# Epoch 6/10
# 112/112 [==============================] - 437s 4s/step - loss: 0.0995 - acc: 0.9659 - val_loss: 0.7826 - val_acc: 0.8259
# Epoch 7/10
# 112/112 [==============================] - 449s 4s/step - loss: 0.1392 - acc: 0.9534 - val_loss: 0.5870 - val_acc: 0.8558
# Epoch 8/10
# 112/112 [==============================] - 463s 4s/step - loss: 0.0986 - acc: 0.9679 - val_loss: 0.5859 - val_acc: 0.8463
# Epoch 9/10
# 112/112 [==============================] - 460s 4s/step - loss: 0.0832 - acc: 0.9693 - val_loss: 0.4346 - val_acc: 0.8966
# Epoch 10/10
# 112/112 [==============================] - 458s 4s/step - loss: 0.0645 - acc: 0.9791 - val_loss: 0.5021 - val_acc: 0.8898
# 2023-05-17 20:24:57.738757: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [895]
# 	 [[{{node Placeholder/_4}}]]
# 2023-05-17 20:24:57.739198: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [895]
# 	 [[{{node Placeholder/_4}}]]
# 5/5 [==============================] - 13s 2s/step - loss: 0.4373 - acc: 0.8875
# [0.4372937083244324, 0.887499988079071]

# 15 epochs, fine_tune_at = 600 , adam , lr scheduler
# 5/5 [==============================] - 13s 3s/step - loss: 0.4462 - acc: 0.9312
# [0.44622063636779785, 0.9312499761581421]

# 112/112 [==============================] - 578s 5s/step - loss: 2.6365 - acc: 0.3052 - val_loss: 1.6575 - val_acc: 0.5306
# Epoch 2/15
# 112/112 [==============================] - 605s 5s/step - loss: 1.2677 - acc: 0.6026 - val_loss: 5.0349 - val_acc: 0.6531
# Epoch 3/15
# 112/112 [==============================] - 596s 5s/step - loss: 0.8585 - acc: 0.7381 - val_loss: 1.1233 - val_acc: 0.7401
# Epoch 4/15
# 112/112 [==============================] - 609s 5s/step - loss: 0.4767 - acc: 0.8512 - val_loss: 3.4846 - val_acc: 0.7578
# Epoch 5/15
# 112/112 [==============================] - 595s 5s/step - loss: 0.2907 - acc: 0.9104 - val_loss: 0.6985 - val_acc: 0.8367
# Epoch 6/15
# 112/112 [==============================] - 568s 5s/step - loss: 0.2353 - acc: 0.9274 - val_loss: 0.8782 - val_acc: 0.8218
# Epoch 7/15
# 112/112 [==============================] - 569s 5s/step - loss: 0.1306 - acc: 0.9573 - val_loss: 0.4162 - val_acc: 0.8816
# Epoch 8/15
# 112/112 [==============================] - 573s 5s/step - loss: 0.0860 - acc: 0.9701 - val_loss: 0.5199 - val_acc: 0.8898
# Epoch 9/15
# 112/112 [==============================] - 571s 5s/step - loss: 0.0834 - acc: 0.9732 - val_loss: 4.3569 - val_acc: 0.8041
# Epoch 10/15
# 112/112 [==============================] - 574s 5s/step - loss: 0.1514 - acc: 0.9497 - val_loss: 0.5451 - val_acc: 0.8776
# Epoch 11/15
# 112/112 [==============================] - 576s 5s/step - loss: 0.1376 - acc: 0.9556 - val_loss: 0.5204 - val_acc: 0.8707
# Epoch 12/15
# 112/112 [==============================] - 575s 5s/step - loss: 0.1072 - acc: 0.9679 - val_loss: 0.4481 - val_acc: 0.8993
# Epoch 13/15
# 112/112 [==============================] - 575s 5s/step - loss: 0.0675 - acc: 0.9807 - val_loss: 0.4917 - val_acc: 0.8939
# Epoch 14/15
# 112/112 [==============================] - 575s 5s/step - loss: 0.0573 - acc: 0.9830 - val_loss: 0.3961 - val_acc: 0.8980
# Epoch 15/15
# 112/112 [==============================] - 574s 5s/step - loss: 0.0222 - acc: 0.9933 - val_loss: 0.3592 - val_acc: 0.9252


# 20 epochs, ResNetRS152, lr=0.01, 700, augmentation,:
# Epoch 1/20
# 112/112 [==============================] - ETA: 0s - loss: 3.1175 - acc: 0.18932023-05-18 11:33:46.439849: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [895]
# 	 [[{{node Placeholder/_4}}]]
# 2023-05-18 11:33:46.440028: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [895]
# 	 [[{{node Placeholder/_4}}]]
# 112/112 [==============================] - 452s 4s/step - loss: 3.1175 - acc: 0.1893 - val_loss: 2.1526 - val_acc: 0.3769
# Epoch 2/20
# 112/112 [==============================] - 467s 4s/step - loss: 1.7752 - acc: 0.4761 - val_loss: 1.6194 - val_acc: 0.5510
# Epoch 3/20
# 112/112 [==============================] - 491s 4s/step - loss: 1.3063 - acc: 0.6012 - val_loss: 1.2080 - val_acc: 0.6408
# Epoch 4/20
# 112/112 [==============================] - 478s 4s/step - loss: 1.0458 - acc: 0.6867 - val_loss: 1.5797 - val_acc: 0.6000
# Epoch 5/20
# 112/112 [==============================] - 489s 4s/step - loss: 0.8592 - acc: 0.7272 - val_loss: 1.0267 - val_acc: 0.6898
# Epoch 6/20
# 112/112 [==============================] - 476s 4s/step - loss: 0.7462 - acc: 0.7685 - val_loss: 0.8662 - val_acc: 0.7483
# Epoch 7/20
# 112/112 [==============================] - 486s 4s/step - loss: 0.6103 - acc: 0.8079 - val_loss: 0.6321 - val_acc: 0.8122
# Epoch 8/20
# 112/112 [==============================] - 498s 4s/step - loss: 0.5383 - acc: 0.8316 - val_loss: 0.8638 - val_acc: 0.7687
# Epoch 9/20
# 112/112 [==============================] - 495s 4s/step - loss: 0.4767 - acc: 0.8637 - val_loss: 0.6527 - val_acc: 0.8340
# Epoch 10/20
# 112/112 [==============================] - 494s 4s/step - loss: 0.4276 - acc: 0.8757 - val_loss: 0.6733 - val_acc: 0.8109
# Epoch 11/20
# 112/112 [==============================] - 500s 4s/step - loss: 0.3725 - acc: 0.8933 - val_loss: 0.5805 - val_acc: 0.8585
# Epoch 12/20
# 112/112 [==============================] - 470s 4s/step - loss: 0.3158 - acc: 0.9039 - val_loss: 0.7729 - val_acc: 0.8245
# Epoch 13/20
# 112/112 [==============================] - 495s 4s/step - loss: 0.2921 - acc: 0.9115 - val_loss: 0.5826 - val_acc: 0.8395
# Epoch 14/20
# 112/112 [==============================] - 472s 4s/step - loss: 0.2526 - acc: 0.9246 - val_loss: 0.4566 - val_acc: 0.8816
# Epoch 15/20
# 112/112 [==============================] - 489s 4s/step - loss: 0.2215 - acc: 0.9338 - val_loss: 0.5613 - val_acc: 0.8776
# Epoch 16/20
# 112/112 [==============================] - 494s 4s/step - loss: 0.2137 - acc: 0.9347 - val_loss: 0.5394 - val_acc: 0.8667
# Epoch 17/20
# 112/112 [==============================] - 482s 4s/step - loss: 0.2187 - acc: 0.9391 - val_loss: 0.5212 - val_acc: 0.8748
# Epoch 18/20
# 112/112 [==============================] - 456s 4s/step - loss: 0.2312 - acc: 0.9330 - val_loss: 0.4726 - val_acc: 0.8707
# Epoch 19/20
# 112/112 [==============================] - 461s 4s/step - loss: 0.1736 - acc: 0.9481 - val_loss: 0.5117 - val_acc: 0.8844
# Epoch 20/20
# 112/112 [==============================] - 450s 4s/step - loss: 0.1809 - acc: 0.9439 - val_loss: 0.4510 - val_acc: 0.9007
# 2023-05-18 14:07:09.816576: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [895]
# 	 [[{{node Placeholder/_0}}]]
# 2023-05-18 14:07:09.816800: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [895]
# 	 [[{{node Placeholder/_4}}]]
# 5/5 [==============================] - 13s 3s/step - loss: 0.2285 - acc: 0.9438
# [0.22845391929149628, 0.9437500238418579]



# Same as above but 30 epochs we get:
# 5/5 [==============================] - 15s 3s/step - loss: 0.5623 - acc: 0.9062
# [0.5623424649238586, 0.90625]
#
# Epoch 25/30
# 112/112 [==============================] - 489s 4s/step - loss: 0.1584 - acc: 0.9534 - val_loss: 0.4160 - val_acc: 0.9075
# Epoch 26/30
# 112/112 [==============================] - 500s 4s/step - loss: 0.1791 - acc: 0.9509 - val_loss: 0.5100 - val_acc: 0.8857
# Epoch 27/30
# 112/112 [==============================] - 514s 5s/step - loss: 0.1703 - acc: 0.9559 - val_loss: 0.3804 - val_acc: 0.9116
# Epoch 28/30
# 112/112 [==============================] - 481s 4s/step - loss: 0.1791 - acc: 0.9495 - val_loss: 0.5123 - val_acc: 0.8884
# Epoch 29/30
# 112/112 [==============================] - 473s 4s/step - loss: 0.1069 - acc: 0.9645 - val_loss: 0.4901 - val_acc: 0.8980
# Epoch 30/30
# 112/112 [==============================] - 581s 5s/step - loss: 0.0919 - acc: 0.9749 - val_loss: 0.5399 - val_acc: 0.8993
#

