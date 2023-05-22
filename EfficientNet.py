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
augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode='horizontal'),
    tf.keras.layers.RandomZoom(0.1),
    #tf.keras.layers.RandomCrop(0.1),
], name='augmentation_layer')
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (augmentation_layer(x), y)) # Where x—images, y—labels.
#val_ds = validation_data.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
#test_ds = test_dataset.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
val_ds = validation_data
test_ds = test_dataset


# STEP 4: Get EfficientNet V2 and ConvNeXtSmall:

# base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(input_shape=(224, 224, 3),
#                                                include_top=False,
#                                                weights='imagenet')

base_model = tf.keras.applications.convnext.ConvNeXtSmall(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')


base_model.trainable = True

fine_tune_at = 250
#total layers: 295 convnext small

print("base model layers", len(base_model.layers))
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
     layer.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


# STEP 5: Generate Model:
model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3), dtype=tf.float32, name='input_image'),
        base_model,
        global_average_layer,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(45, dtype=tf.float32, activation='softmax')
    ])

model.summary()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)


# STEP 6: Compile the model:
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics='accuracy'
)
# STEP 7: Fit the model:
history = model.fit(train_ds,
                epochs=20,
                validation_data=val_ds,
                validation_steps=int(len(val_ds)),
                shuffle=False)


# STEP 8: Evaluate:
print(model.evaluate(test_ds))


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

# lr=0.001
# After 10 epochs:  loss: 0.3545 - acc: 0.8822 - val_loss: 0.7879 - val_acc: 0.7911

print(model.evaluate(test_ds))



# EfficientNet, Epoch 20/20 lr=0.001:

# 112/112 [==============================]
# - 35s 312ms/step - loss: 0.1299 - acc: 0.9542 - val_loss: 0.6327 - val_acc: 0.8146

# After 20 epochs on test: lr=0.001
# 2/2 [==============================] - 1s 290ms/step - loss: 0.6322 - acc: 0.7812
# Test Loss and Acc: [0.6321812272071838, 0.78125]

#------------------------------------------------------------------------------------
# After 30 epochs on test: lr=0.001:
# Epoch 30/30
# 112/112 [==============================] - 36s 325ms/step - loss: 0.1117 - acc: 0.9615 - val_loss: 0.6393 - val_acc: 0.8433
# 2/2 [==============================] - 1s 291ms/step - loss: 0.2503 - acc: 0.9062
# Test Loss and Acc: [0.2502741813659668, 0.90625]



# 50 epochs, 320, EfficientNetV2B1, augmentation layer, 0.0001
# 5/5 [==============================] - 2s 446ms/step - loss: 0.2520 - accuracy: 0.9062
# Test Loss and Acc: [0.2519591450691223, 0.90625]


# 80 epochs, EfficientNetV2B1, augmentation layer, 0.0001:


# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
# Total
# params: 7729889(29.49
# MB)
# Trainable
# params: 2186125(8.34
# MB)
# Non - trainable
# params: 5543764(21.15
# MB)
# _________________________________________________________________
#
#
# Epoch 1 / 80
# 112 / 112[== == == == == == == == == == == == == == ==] - ETA: 0
# s - loss: 3.2596 - accuracy: 0.21312023 - 05 - 18
# 22: 48:54.903191:
# 112 / 112[== == == == == == == == == == == == == == ==] - 71s 606ms / step - loss: 3.2596 - accuracy: 0.2131 - val_loss: 2.5043 - val_accuracy: 0.4027
# Epoch
# 2 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 69
# s
# 612
# ms / step - loss: 2.2478 - accuracy: 0.4437 - val_loss: 1.7447 - val_accuracy: 0.5673
# Epoch
# 3 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 67
# s
# 593
# ms / step - loss: 1.6576 - accuracy: 0.5658 - val_loss: 1.3488 - val_accuracy: 0.6340
# Epoch
# 4 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 67
# s
# 594
# ms / step - loss: 1.3028 - accuracy: 0.6342 - val_loss: 1.0969 - val_accuracy: 0.6939
# Epoch
# 5 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 66
# s
# 585
# ms / step - loss: 1.0917 - accuracy: 0.6981 - val_loss: 0.9751 - val_accuracy: 0.7034
# Epoch
# 6 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 66
# s
# 593
# ms / step - loss: 0.9318 - accuracy: 0.7397 - val_loss: 0.8384 - val_accuracy: 0.7429
# Epoch
# 7 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 67
# s
# 597
# ms / step - loss: 0.8032 - accuracy: 0.7766 - val_loss: 0.7569 - val_accuracy: 0.7810
# Epoch
# 8 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 71
# s
# 632
# ms / step - loss: 0.7010 - accuracy: 0.8040 - val_loss: 0.6605 - val_accuracy: 0.8082
# Epoch
# 9 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 69
# s
# 615
# ms / step - loss: 0.6150 - accuracy: 0.8355 - val_loss: 0.6087 - val_accuracy: 0.8122
# Epoch
# 10 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 65
# s
# 583
# ms / step - loss: 0.5438 - accuracy: 0.8447 - val_loss: 0.5973 - val_accuracy: 0.8122
# Epoch
# 11 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 67
# s
# 600
# ms / step - loss: 0.4945 - accuracy: 0.8576 - val_loss: 0.5249 - val_accuracy: 0.8476
# Epoch
# 12 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 65
# s
# 582
# ms / step - loss: 0.4417 - accuracy: 0.8771 - val_loss: 0.5026 - val_accuracy: 0.8476
# Epoch
# 13 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 66
# s
# 587
# ms / step - loss: 0.3965 - accuracy: 0.8869 - val_loss: 0.4429 - val_accuracy: 0.8612
# Epoch
# 14 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 571
# ms / step - loss: 0.3674 - accuracy: 0.8936 - val_loss: 0.4604 - val_accuracy: 0.8653
# Epoch
# 15 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 66
# s
# 589
# ms / step - loss: 0.3244 - accuracy: 0.9076 - val_loss: 0.4323 - val_accuracy: 0.8639
# Epoch
# 16 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 67
# s
# 602
# ms / step - loss: 0.2961 - accuracy: 0.9201 - val_loss: 0.4190 - val_accuracy: 0.8748
# Epoch
# 17 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 65
# s
# 581
# ms / step - loss: 0.2722 - accuracy: 0.9257 - val_loss: 0.3846 - val_accuracy: 0.8884
# Epoch
# 18 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 65
# s
# 578
# ms / step - loss: 0.2585 - accuracy: 0.9257 - val_loss: 0.3848 - val_accuracy: 0.8816
# Epoch
# 19 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 65
# s
# 581
# ms / step - loss: 0.2294 - accuracy: 0.9380 - val_loss: 0.3611 - val_accuracy: 0.8884
# Epoch
# 20 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 65
# s
# 582
# ms / step - loss: 0.2195 - accuracy: 0.9333 - val_loss: 0.3550 - val_accuracy: 0.8871
# Epoch
# 21 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 575
# ms / step - loss: 0.2095 - accuracy: 0.9408 - val_loss: 0.3774 - val_accuracy: 0.8816
# Epoch
# 22 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 573
# ms / step - loss: 0.2082 - accuracy: 0.9380 - val_loss: 0.3502 - val_accuracy: 0.8966
# Epoch
# 23 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 574
# ms / step - loss: 0.1922 - accuracy: 0.9439 - val_loss: 0.3082 - val_accuracy: 0.9048
# Epoch
# 24 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 65
# s
# 576
# ms / step - loss: 0.1934 - accuracy: 0.9481 - val_loss: 0.3129 - val_accuracy: 0.8952
# Epoch
# 25 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 67
# s
# 595
# ms / step - loss: 0.1783 - accuracy: 0.9458 - val_loss: 0.3151 - val_accuracy: 0.9075
# Epoch
# 26 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 65
# s
# 583
# ms / step - loss: 0.1525 - accuracy: 0.9567 - val_loss: 0.3504 - val_accuracy: 0.8993
# Epoch
# 27 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 67
# s
# 595
# ms / step - loss: 0.1580 - accuracy: 0.9573 - val_loss: 0.3317 - val_accuracy: 0.9048
# Epoch
# 28 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 574
# ms / step - loss: 0.1426 - accuracy: 0.9612 - val_loss: 0.2952 - val_accuracy: 0.9061
# Epoch
# 29 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 66
# s
# 594
# ms / step - loss: 0.1369 - accuracy: 0.9587 - val_loss: 0.3184 - val_accuracy: 0.8980
# Epoch
# 30 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 65
# s
# 579
# ms / step - loss: 0.1359 - accuracy: 0.9626 - val_loss: 0.2906 - val_accuracy: 0.9075
# Epoch
# 31 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 572
# ms / step - loss: 0.1405 - accuracy: 0.9601 - val_loss: 0.3353 - val_accuracy: 0.9034
# Epoch
# 32 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 569
# ms / step - loss: 0.1348 - accuracy: 0.9559 - val_loss: 0.3352 - val_accuracy: 0.9061
# Epoch
# 33 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 62
# s
# 549
# ms / step - loss: 0.1246 - accuracy: 0.9657 - val_loss: 0.2867 - val_accuracy: 0.9143
# Epoch
# 34 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 63
# s
# 559
# ms / step - loss: 0.1239 - accuracy: 0.9634 - val_loss: 0.3065 - val_accuracy: 0.9007
# Epoch
# 35 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 62
# s
# 555
# ms / step - loss: 0.1118 - accuracy: 0.9698 - val_loss: 0.3189 - val_accuracy: 0.9102
# Epoch
# 36 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 548
# ms / step - loss: 0.1161 - accuracy: 0.9657 - val_loss: 0.2817 - val_accuracy: 0.9156
# Epoch
# 37 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 548
# ms / step - loss: 0.1156 - accuracy: 0.9634 - val_loss: 0.2549 - val_accuracy: 0.9306
# Epoch
# 38 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 547
# ms / step - loss: 0.1160 - accuracy: 0.9645 - val_loss: 0.3119 - val_accuracy: 0.9088
# Epoch
# 39 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 568
# ms / step - loss: 0.0938 - accuracy: 0.9740 - val_loss: 0.2777 - val_accuracy: 0.9156
# Epoch
# 40 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 543
# ms / step - loss: 0.0981 - accuracy: 0.9721 - val_loss: 0.2709 - val_accuracy: 0.9211
# Epoch
# 41 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 547
# ms / step - loss: 0.1034 - accuracy: 0.9704 - val_loss: 0.2514 - val_accuracy: 0.9293
# Epoch
# 42 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 543
# ms / step - loss: 0.0847 - accuracy: 0.9751 - val_loss: 0.2535 - val_accuracy: 0.9252
# Epoch
# 43 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 544
# ms / step - loss: 0.0874 - accuracy: 0.9743 - val_loss: 0.2856 - val_accuracy: 0.9265
# Epoch
# 44 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 62
# s
# 554
# ms / step - loss: 0.0912 - accuracy: 0.9740 - val_loss: 0.2649 - val_accuracy: 0.9293
# Epoch
# 45 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 545
# ms / step - loss: 0.0931 - accuracy: 0.9743 - val_loss: 0.2595 - val_accuracy: 0.9293
# Epoch
# 46 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 62
# s
# 550
# ms / step - loss: 0.0801 - accuracy: 0.9754 - val_loss: 0.2525 - val_accuracy: 0.9279
# Epoch
# 47 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 545
# ms / step - loss: 0.0805 - accuracy: 0.9757 - val_loss: 0.2241 - val_accuracy: 0.9293
# Epoch
# 48 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 548
# ms / step - loss: 0.0779 - accuracy: 0.9785 - val_loss: 0.2652 - val_accuracy: 0.9211
# Epoch
# 49 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 62
# s
# 554
# ms / step - loss: 0.0688 - accuracy: 0.9796 - val_loss: 0.2493 - val_accuracy: 0.9306
# Epoch
# 50 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 62
# s
# 551
# ms / step - loss: 0.0671 - accuracy: 0.9799 - val_loss: 0.2764 - val_accuracy: 0.9170
# Epoch
# 51 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 547
# ms / step - loss: 0.0893 - accuracy: 0.9710 - val_loss: 0.2356 - val_accuracy: 0.9442
# Epoch
# 52 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 549
# ms / step - loss: 0.0755 - accuracy: 0.9774 - val_loss: 0.2789 - val_accuracy: 0.9293
# Epoch
# 53 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 549
# ms / step - loss: 0.0664 - accuracy: 0.9807 - val_loss: 0.2925 - val_accuracy: 0.9320
# Epoch
# 54 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 63
# s
# 561
# ms / step - loss: 0.0642 - accuracy: 0.9816 - val_loss: 0.2833 - val_accuracy: 0.9170
# Epoch
# 55 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 62
# s
# 553
# ms / step - loss: 0.0703 - accuracy: 0.9777 - val_loss: 0.2923 - val_accuracy: 0.9252
# Epoch
# 56 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 62
# s
# 552
# ms / step - loss: 0.0721 - accuracy: 0.9777 - val_loss: 0.2552 - val_accuracy: 0.9320
# Epoch
# 57 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 63
# s
# 560
# ms / step - loss: 0.0687 - accuracy: 0.9774 - val_loss: 0.2413 - val_accuracy: 0.9279
# Epoch
# 58 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 62
# s
# 555
# ms / step - loss: 0.0736 - accuracy: 0.9791 - val_loss: 0.2265 - val_accuracy: 0.9429
# Epoch
# 59 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 547
# ms / step - loss: 0.0561 - accuracy: 0.9830 - val_loss: 0.2391 - val_accuracy: 0.9347
# Epoch
# 60 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 544
# ms / step - loss: 0.0570 - accuracy: 0.9818 - val_loss: 0.2169 - val_accuracy: 0.9401
# Epoch
# 61 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 68
# s
# 612
# ms / step - loss: 0.0547 - accuracy: 0.9832 - val_loss: 0.2606 - val_accuracy: 0.9306
# Epoch
# 62 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 66
# s
# 589
# ms / step - loss: 0.0624 - accuracy: 0.9824 - val_loss: 0.2477 - val_accuracy: 0.9306
# Epoch
# 63 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 66
# s
# 587
# ms / step - loss: 0.0613 - accuracy: 0.9796 - val_loss: 0.2528 - val_accuracy: 0.9333
# Epoch
# 64 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 574
# ms / step - loss: 0.0589 - accuracy: 0.9805 - val_loss: 0.2653 - val_accuracy: 0.9333
# Epoch
# 65 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 66
# s
# 588
# ms / step - loss: 0.0597 - accuracy: 0.9844 - val_loss: 0.2364 - val_accuracy: 0.9415
# Epoch
# 66 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 66
# s
# 587
# ms / step - loss: 0.0531 - accuracy: 0.9844 - val_loss: 0.2407 - val_accuracy: 0.9401
# Epoch
# 67 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 568
# ms / step - loss: 0.0526 - accuracy: 0.9855 - val_loss: 0.2709 - val_accuracy: 0.9211
# Epoch
# 68 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 66
# s
# 590
# ms / step - loss: 0.0531 - accuracy: 0.9846 - val_loss: 0.2704 - val_accuracy: 0.9252
# Epoch
# 69 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 63
# s
# 563
# ms / step - loss: 0.0498 - accuracy: 0.9858 - val_loss: 0.2389 - val_accuracy: 0.9306
# Epoch
# 70 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 72
# s
# 640
# ms / step - loss: 0.0524 - accuracy: 0.9841 - val_loss: 0.2358 - val_accuracy: 0.9320
# Epoch
# 71 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 66
# s
# 592
# ms / step - loss: 0.0527 - accuracy: 0.9846 - val_loss: 0.2258 - val_accuracy: 0.9347
# Epoch
# 72 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 69
# s
# 614
# ms / step - loss: 0.0556 - accuracy: 0.9830 - val_loss: 0.2414 - val_accuracy: 0.9306
# Epoch
# 73 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 574
# ms / step - loss: 0.0506 - accuracy: 0.9858 - val_loss: 0.2656 - val_accuracy: 0.9293
# Epoch
# 74 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 574
# ms / step - loss: 0.0543 - accuracy: 0.9830 - val_loss: 0.2462 - val_accuracy: 0.9388
# Epoch
# 75 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 63
# s
# 566
# ms / step - loss: 0.0435 - accuracy: 0.9855 - val_loss: 0.2314 - val_accuracy: 0.9401
# Epoch
# 76 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 569
# ms / step - loss: 0.0455 - accuracy: 0.9844 - val_loss: 0.2484 - val_accuracy: 0.9333
# Epoch
# 77 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 575
# ms / step - loss: 0.0454 - accuracy: 0.9844 - val_loss: 0.2268 - val_accuracy: 0.9388
# Epoch
# 78 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 62
# s
# 557
# ms / step - loss: 0.0500 - accuracy: 0.9838 - val_loss: 0.2230 - val_accuracy: 0.9388
# Epoch
# 79 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 61
# s
# 547
# ms / step - loss: 0.0523 - accuracy: 0.9844 - val_loss: 0.2495 - val_accuracy: 0.9333
# Epoch
# 80 / 80
# 112 / 112[ == == == == == == == == == == == == == == ==] - 64
# s
# 572
# ms / step - loss: 0.0491 - accuracy: 0.9855 - val_loss: 0.2354 - val_accuracy: 0.9429
#
# 5 / 5[== == == == == == == == == == == == == == ==] - 2s 420ms / step - loss: 0.1591 - accuracy: 0.9625
# Test Loss and Acc: [0.15908123552799225, 0.9624999761581421]


# Convnext 0.0001, 10 epochs:
# fine_tune_at = 250
# #295 convnext small
# 5/5 [==============================] - 49s 10s/step - loss: 0.3547 - accuracy: 0.8938
# [0.3547435402870178, 0.893750011920929]

# 20 epochs, lr scheduler, same as above:
#
# 5/5 [==============================] - 48s 10s/step - loss: 0.3464 - accuracy: 0.8938
# Test Loss and Acc: [0.34639063477516174, 0.893750011920929]