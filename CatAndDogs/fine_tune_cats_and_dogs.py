import os

from datetime import datetime

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from typing import *

import argparse
from CatsAndDogsClassifier import CatsAndDogsClassifier

parser = argparse.ArgumentParser('Training and fine-tuning a resnet for the Cats and Dogs dataset')
parser.add_argument('model_type', type=str, default='Binary', help='Type of model that you want to fine-tune. '
                                                                   'Either \'Binary\' or \'Multi\'')
parser.add_argument('path_to_training_images', type=str, help='Path to the training images')
parser.add_argument('path_to_validation_images', type=str, help='Path to the training images')
parser.add_argument('path_to_training_pkl', type=str, help='Path to the training dataframe.')
parser.add_argument('path_to_validation_pkl', type=str, help='Path to the validation dataframe.')
parser.add_argument('path_to_save_models', type=str, help='Path pointing to the location where the models should be saved to.')
parser.add_argument('path_to_save_history', type=str, help='Path pointing to the location where the history should be saved to.')

parser.add_argument('initial_epochs', type=int, default=10, help='Number of epochs to train just the top layer')
parser.add_argument('fine_tuning_epochs', type=int, default=10, help='Number of epochs to finetune multiple pretrained layers.')
parser.add_argument('lr', type=float, default=0.0001, help='Base learning rate for the Adams optimizer.')
parser.add_argument('batch_size', type=int, default=64, help='Batch size to use for both training and validation')
parser.add_argument('fine_tune_at', type=int, default=30, help='From which layer onwards the weights should be unfrozen.')
parser.add_argument('--augment', action='store_true', help='Indicates if data augmentation should be used.')
parser.add_argument('--lr_scheduler', action='store_true', help='Use learning rate scheduler')


def read_image_data_augmentation(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
    if tf.random.uniform(shape=[]) > 0.5:
        image = tf.image.flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=0.05)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    image = tf.image.resize_with_crop_or_pad(image, target_width=224, target_height=224)
    image = tf.image.per_image_standardization(image)
    return image, label


def read_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, target_width=224, target_height=224)
    image = tf.image.per_image_standardization(image)
    return image, label


def scheduler(epoch: int, lr: float) -> float:
    if epoch < 10:
        return lr
    else:
        return lr*tf.math.exp(-0.1)


def load_dataset(path_to_images: str,
                 path_to_pkl: str,
                 model_type: str,
                 batch_size: int,
                 data_type: str = 'train',
                 data_augmentation: bool = False) -> tf.data.Dataset:
    df = pd.read_pickle(path_to_pkl)
    df['file_name'] = path_to_images + df['file_name'] + '.jpg'

    x = df['file_name'].values
    if model_type == "Binary":
        initial_values = df['species'].values - 1
        y = np.zeros((len(initial_values), 2), dtype=int)

        for i, val in enumerate(initial_values):
            y[i, val] = 1
    else:
        y = np.stack(df['id_onehot'].values)

    if data_augmentation and data_type == 'train':
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(read_image_data_augmentation).batch(batch_size=batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(read_image).batch(batch_size=batch_size)
    return dataset


def save_history(history, path: str):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    history_dict = history.history
    # Save it under the form of a json file
    s = str(history_dict)
    with open(path, 'w') as file:
        file.write(s)


def plot_loss_acc(history, path: str, model_type: str):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    fig = plt.figure(figsize=(8, 8))

    if model_type == 'Binary':
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        plt.subplot(2, 1, 1)
        plt.plot(train_acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')
    else:
        train_auc = history.history['auc']
        val_auc = history.history['val_auc']

        plt.subplot(2, 1, 1)
        plt.plot(train_auc, label='Training AUC')
        plt.plot(val_auc, label='Validation AUC')
        plt.legend(loc='lower right')
        plt.ylabel('AUC')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation AUC')

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(2, 1, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def main():
    args = parser.parse_args()
    model_type: str = args.model_type

    path_to_training_images: str = args.path_to_training_images
    path_to_validation_images: str = args.path_to_validation_images
    path_to_training_pkl: str = args.path_to_training_pkl
    path_to_validation_pkl: str = args.path_to_validation_pkl
    path_to_save_models: str = args.path_to_save_models
    path_to_save_history: str = args.path_to_save_history

    initial_epochs: int = args.initial_epochs
    fine_tuning_epochs: int = args.fine_tuning_epochs
    lr: float = args.lr
    batch_size: int = args.batch_size
    fine_tune_at: int = args.fine_tune_at
    augment: bool = args.augment
    lr_scheduler: bool = args.lr_scheduler

    if model_type not in ['Binary', 'Multi']:
        raise ValueError(f'Model type \'{model_type}\' was not recognized. It has to be either \'Binary\' or \'Multi\'')

    if not os.path.exists(path_to_training_images):
        raise IOError(f'Path to training images does not exist! \'{path_to_training_images}\'')

    if not os.path.exists(path_to_validation_images):
        raise IOError(f'Path to validation images does not exist! \'{path_to_validation_images}\'')

    if not os.path.exists(path_to_training_pkl):
        raise IOError(f'Path to training csv does not exist! \'{path_to_training_pkl}\'')

    if not os.path.exists(path_to_validation_pkl):
        raise IOError(f'Path to validation csv does not exist! \'{path_to_validation_pkl}\'')

    if not os.path.exists(path_to_save_models):
        os.makedirs(path_to_save_models)


    training_dataset: tf.data.Dataset = load_dataset(path_to_images=path_to_training_images,
                                                     path_to_pkl=path_to_training_pkl,
                                                     model_type=model_type,
                                                     batch_size=batch_size,
                                                     data_type='train',
                                                     data_augmentation=augment)
    validation_dataset: tf.data.Dataset = load_dataset(path_to_images=path_to_validation_images,
                                                       path_to_pkl=path_to_validation_pkl,
                                                       model_type=model_type,
                                                       batch_size=batch_size,
                                                       data_type='val')

    number_of_output_classes: int = 2 if model_type == 'Binary' else 37

    loss: tf.keras.losses
    metric: List[tf.keras.metrics] = ['accuracy']
    if model_type == 'Binary':
        loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        loss = tf.keras.losses.CategoricalCrossentropy()

    callbacks = []
    lr_sched = lr

    monitor: str = 'val_accuracy'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(path_to_save_models, 'checkpoints'),
        save_weights_only=True,
        monitor=monitor,
        mode='max',
        save_best_only=True)
    callbacks.append(model_checkpoint_callback)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7)
    callbacks.append(early_stopping)

    cats_and_dogs_classifier: CatsAndDogsClassifier = CatsAndDogsClassifier(number_of_output_classes=number_of_output_classes)
    cats_and_dogs_classifier.compile(optimizer=tf.keras.optimizers.Adam(lr_sched),
                                     loss=loss,
                                     metrics=metric)

    training_history = cats_and_dogs_classifier.fit(training_dataset,
                                                    epochs=initial_epochs,
                                                    validation_data=validation_dataset,
                                                    callbacks=callbacks)

    datetime_str: str = datetime.now().strftime('%Y%H%M%S')
    # plot_loss_acc(training_history, os.path.join(path_to_save_history, f'{datetime_str}_model_{model_type}_initial.png'), model_type=model_type)
    save_history(training_history, path=os.path.join(path_to_save_history, f'{datetime_str}_model_{model_type}_initial.json'))
    cats_and_dogs_classifier.save_weights(os.path.join(path_to_save_models, f'{datetime_str}_model_{model_type}_initial', f'model'))

    cats_and_dogs_classifier.unfreeze_top_layers(fine_tune_top_n=fine_tune_at)

    if lr_scheduler:
        lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                                                                  decay_rate=0.9,
                                                                  decay_steps=500)

    cats_and_dogs_classifier.compile(optimizer=tf.keras.optimizers.Adam(lr_sched),
                                     loss=loss,
                                     metrics=metric)

    total_epochs = initial_epochs + fine_tuning_epochs

    fine_tuning_history = cats_and_dogs_classifier.fit(training_dataset,
                                                       epochs=total_epochs,
                                                       initial_epoch=training_history.epoch[-1],
                                                       validation_data=validation_dataset,
                                                       callbacks=callbacks)

    datetime_str: str = datetime.now().strftime('%Y%H%M%S')
    #plot_loss_acc(fine_tuning_history, os.path.join(path_to_save_history, f'{datetime_str}_model_{model_type}_{fine_tune_at}_finetuned.png'), model_type=model_type)
    save_history(fine_tuning_history, path=os.path.join(path_to_save_history, f'{datetime_str}_model_{model_type}_{fine_tune_at}_finetuned.json'))
    cats_and_dogs_classifier.save_weights(os.path.join(path_to_save_models, f'{datetime_str}_model_{model_type}_{fine_tune_at}_finetuned', 'model'))


if __name__ == '__main__':
    main()