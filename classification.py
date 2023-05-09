import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
#import tf.models as tfm
#import tensorflow_models as tfm


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

    # Make sure the folder contains a file for each of the 44 categories:
    for file in filenames:
        new_folder_path = os.path.join(new_path, file)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

    return




# Run the following lines once (alternatively clear files for different runs/draws of data splits):
#create_files(train, 'training_data/')
#create_files(val, 'validation_data/')


# This is to check if a file is empty or not incase we want to either repopulate it or delete its contents for new runs
# print(os.path.getsize(new_path) == 0)
# print(len(os.listdir(new_path)))






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

# This tries to imitate the way tf builds classes [not 100% sure it is correct]

def get_classes(folder_path):
    filenames = os.listdir(folder_path)
    sorted_files = sorted(filenames)
    # This is to exclude: '.DS_Store':
    return [f for f in sorted_files if not f.startswith('.')]
    #return sorted(filenames)

print(get_classes('training_data/'))

class_names = get_classes('training_data/')
# Seems like this does the trick (for some reason, it was not working)
class_names = np.array(train_ds.class_names)
print("class names", class_names)

# Show the first nine images and labels from the training set:
## following steps in: https://www.tensorflow.org/tutorials/images/transfer_learning

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         #print("labels", class_names[tf.math.argmax(labels[i])])
#         plt.title(class_names[tf.math.argmax(labels[i])])
#         plt.axis("off")
# plt.show()


# Create test data:
val_batches = tf.data.experimental.cardinality(validation_ds)
print('Number of val batches: %d' % val_batches)
test_dataset = validation_ds.take(val_batches // 5)
validation_data = validation_ds.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_data))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


# NORMALIZE DATA:

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
val_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Download the classifer:

mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
resnet50 = "https://tfhub.dev/tensorflow/resnet_50/classification/1"
classifier_model = resnet50

IMAGE_SHAPE = (224, 224)
import tensorflow_hub as hub


# Download the headless model:

feature_extractor_model = resnet50

feature_extractor_layer = hub.KerasLayer(
feature_extractor_model,
input_shape=(224, 224, 3),
trainable=False)

# for image_batch, labels_batch in train_ds:
#
#   feature_batch = feature_extractor_layer(image_batch)
#

# Attach a classification head:
num_classes = len(class_names)

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])

model.summary()

# for image_batch, labels_batch in train_ds:
#     predictions = model(image_batch)
#     print("predictions shape", predictions.shape)
#     break

# Train the model:

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
     loss='categorical_crossentropy',
  metrics=['acc'])

NUM_EPOCHS = 10

history = model.fit(train_ds,
                   validation_data=val_ds,
                    epochs=NUM_EPOCHS)








# To load a single batch:
# for image_batch, labels_batch in train_ds:
#     x_train = image_batch.numpy()
#     y_train = labels_batch.numpy()
#     break
#
# for image_batch, labels_batch in validation_ds:
#     x_val = image_batch.numpy()
#     y_val = labels_batch.numpy()
#     break
#
# print("x_train", len(x_train))
# print("y_train", len(y_train))
#
# print("x_train", x_train[1].shape)
#print("y_train", y_train.shape())




# USE https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub

# useful tutorial, another model: https://www.kaggle.com/code/bulentsiyah/dogs-vs-cats-classification-vgg16-fine-tuning

# Tutorial using pretrained convnext
# https://www.kaggle.com/code/shrijeethsuresh/brain-tumor-convnext-44-classes-99-4-acc

# Tutorial with transfer learning on tumor data: To follow general steps in:
# https://www.kaggle.com/code/sanandachowdhury/transfer-learning-brain-tumor-classification