import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
#import tf.models as tfm
#import tensorflow_models as tfm
import tensorflow_hub as hub
import vit_keras
from vit_keras import vit

from transformers import ViTFeatureExtractor, Trainer, TrainingArguments
from transformers import ViTForImageClassification


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


# CREATE FIRST FOLDERS: training_data and validation_data:

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


class_names = np.array(train_ds.class_names)
print("class names", class_names)



# STEP 2: Create test data:
val_batches = tf.data.experimental.cardinality(validation_ds)
print('Number of val batches: %d' % val_batches)
test_dataset = validation_ds.take(val_batches // 5)
validation_data = validation_ds.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_data))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

# Build Augmentation layer:

augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode='horizontal')
], name='augmentation_layer')


# STEP 3: Normalize data:
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (augmentation_layer(normalization_layer(x)), y)) # Where x—images, y—labels.
val_ds = validation_data.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
test_ds = test_dataset.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.


AUTOTUNE = tf.data.AUTOTUNE
#train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# STEP 4: Get pre-trained models from this link:

# Download the model
# vit_model = vit_keras.vit.vit_b16(
#         image_size=224,
#         activation='softmax',
#         pretrained=True,
#         include_top=False,
#         pretrained_top=False,
#         classes=2)

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)


model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(class_names),
    id2label={str(i): c for i, c in enumerate(class_names)},
    label2id={c: str(i) for i, c in enumerate(class_names)}
)

training_args = TrainingArguments(
  output_dir="./vit-base-beans",
  per_device_train_batch_size=32,
  evaluation_strategy="steps",
  num_train_epochs=5,
  fp16=True,
  # save_steps=100,
  eval_steps=100,
  # logging_steps=10,
  learning_rate=1e-3,
  # save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  # report_to='tensorboard',
  load_best_model_at_end=True,
)








trainer = Trainer(
    model=model,
    args=training_args,
    # data_collator=collate_fn,
    # compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=feature_extractor,
)

train_results = trainer.train()
print(train_results.metrics)

metrics = trainer.evaluate(val_ds)





#
#
# # Freeze model layers for inference-mode only
# for layer in vit_model.layers:
#     layer.trainable = False
#
# # STEP 5: Build Model:
# num_classes = len(class_names)
# initializer = tf.keras.initializers.GlorotNormal(seed=31)
# model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(224, 224, 3), dtype=tf.float32, name='input_image'),
#         vit_model,
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(512, activation='relu',  kernel_initializer=initializer),
#         tf.keras.layers.Dense(256, activation='relu',  kernel_initializer=initializer),
#         tf.keras.layers.Dense(44, dtype=tf.float32, activation='softmax', kernel_initializer=initializer)
#   ])
#
# model.summary()
#
# # STEP 6: Compile the model:
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=10000,
#     decay_rate=0.9)
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#      #loss='categorical_crossentropy',
#     loss=tf.keras.losses.CategoricalCrossentropy(),
#     metrics=['acc'])
#
# NUM_EPOCHS = 20
#
# # STEP 7: Fit the model:
# history = model.fit(train_ds,
#                     validation_data=val_ds,
#                     epochs=NUM_EPOCHS)
#
# # STEP 8: Evaluate:
# print(model.evaluate(test_ds))
#
# #lr=0.01
# # Epoch 20/20
# # 112/112 [==============================] - 130s 1s/step - loss: 0.7740 - acc: 0.7400 - val_loss: 1.7613 - val_acc: 0.6005
# # Test loss and acc: [2.276601552963257, 0.578125]
#
# # 40 epochs: lr=0.01
# # Epoch 40/40
# # 112/112 [==============================] - 137s 1s/step - loss: 0.4745 - acc: 0.8419 - val_loss: 2.1517 - val_acc: 0.6188
# # Test: [1.7676501274108887, 0.59375]
#
#
#
# # STEP 9: Plot loss and accuracies:
# from matplotlib import pyplot as plt
#
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
#
#
# https://huggingface.co/blog/fine-tune-vit