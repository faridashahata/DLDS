import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2

#print(os.listdir())

data_list = []
normal_list = []
with open('data_list.txt', 'r') as f:
    for line in f:
        image_list = line.split(";")
        if "_NORMAL" in image_list[1]:
            normal_list.append([image_list[0], image_list[1].replace('\n', '')])
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

    val_list_unnormal = data_list[train_lim_unnormal:train_lim_unnormal + val_lim_unnormal]
    val_list_normal = normal_list[train_lim_normal:train_lim_normal + val_lim_normal]
    val_list = [*val_list_unnormal, *val_list_normal]

    test_list_unnormal = data_list[train_lim_unnormal + val_lim_unnormal:]
    test_list_normal = normal_list[train_lim_normal + val_lim_normal:]
    test_list = [*test_list_unnormal, *test_list_normal]

    return train_list, val_list, test_list


train, val, test = store_train_val(data_list, normal_list, 0.8, 0.1, 0.1)



print("len of train", len(train))
print("len of val", len(val))
print("len of test", len(test))


# CREATE FIRST FOLDERS: training_data and validation data:

def create_files(file_names, folder_path):
    path_extract = "resized_images/"
    filenames = os.listdir(path_extract)
    #new_path = 'training_data/'
    new_path = folder_path

    #instead of file_names, put train, val or test:
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

    return




# create_files(train, 'training_data/')
# create_files(val, 'validation_data/')
#

# This is to check if a file is empty or not incase we want to either repopulate it or delete its contents for new runs
# print(os.path.getsize(new_path) == 0)
# print(len(os.listdir(new_path)))






# Apparently tf creates classes in an alphanumeric fashion.
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

# This tries to imitate the way tf builds classes [not 100% sure it is correct]

def get_classes(folder_path):
    filenames = os.listdir(folder_path)
    sorted_files = sorted(filenames)
    # This is to exclude: '.DS_Store':
    return [f for f in sorted_files if not f.startswith('.')]
    #return sorted(filenames)

print(get_classes('training_data/'))

class_names = get_classes('training_data/')


# Show the first nine images and labels from the training set:
## following steps in: https://www.tensorflow.org/tutorials/images/transfer_learning

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        #print("labels", class_names[tf.math.argmax(labels[i])])
        plt.title(class_names[tf.math.argmax(labels[i])])
        plt.axis("off")
plt.show()

