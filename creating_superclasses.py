import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
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


#train, val, test = store_train_val(data_list, normal_list, 0.8, 0.2, 0.0)
#
#
# print("len of train", len(train))
# print("len of val", len(val))
# print("len of test", len(test))


# CREATE FIRST FOLDERS: training_data_superclasses and validation_data_superclasses:

def create_files(file_names, folder_path):
    path_extract = "resized_images/"
    filenames = os.listdir(path_extract)
    #new_path = 'training_data/'
    new_path = folder_path

    # instead of file_names, put train, val or test:
    for image in file_names:
        image_name = image[0]
        image_class = image[1]
        image_superclass = image_class.split()[0]
        print("image_class", image_class)

        im_path = os.path.join(path_extract, image_class, image_name)
        new_folder_path = os.path.join(new_path, image_superclass)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        new_im_path = os.path.join(new_folder_path, image_name)
        image_file = cv2.imread(im_path)
        cv2.imwrite(new_im_path, image_file)

    # Make sure the folder contains a file for each of the 44 categories:
    # for file in filenames:
    #     new_folder_path = os.path.join(new_path, file)
    #     if not os.path.exists(new_folder_path):
    #         os.makedirs(new_folder_path)

    return

# Run this on a newly created training data folder:

# # Run the following lines once (alternatively clear files for different runs/draws of data splits):
# create_files(train, 'training_data_superclasses/')
# create_files(val, 'validation_data_superclasses/')

#
# STEP 1: Read data from directory:
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='training_data_superclasses/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224))


validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory='validation_data_superclasses/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224))


class_names = np.array(train_ds.class_names)
print("class names", class_names)
