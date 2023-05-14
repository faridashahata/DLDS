import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from typing import *

from datetime import datetime

import argparse
from NiHClassifier import NiHClassifier

parser = argparse.ArgumentParser('Training and fine-tuning a resnet for the NiH Lung dataset')
parser.add_argument('model_type', type=str, default='Binary', help='Type of model that you want to fine-tune. '
                                                                   'Either \'Binary\' or \'Multi\'')
parser.add_argument('path_to_training_images', type=str, help='Path to the training images')
parser.add_argument('path_to_validation_images', type=str, help='Path to the training images')
parser.add_argument('path_to_training_pkl', type=str, help='Path to the training dataframe.')
parser.add_argument('path_to_validation_pkl', type=str, help='Path to the validation dataframe.')
parser.add_argument('path_to_save_models', type=str, help='Path pointing to the location where the models should be saved to.')
parser.add_argument('path_to_figures', type=str, help='Path pointing to the location where the figures should be saved to.')

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
    return image, label


def read_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
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
                 data_augmentation: bool = False) -> tf.data.Dataset:
    df = pd.read_pickle(path_to_pkl)
    df['Image Index'] = path_to_images + df['Image Index']

    x = df['Image Index'].values
    if model_type == "Binary":
        y = df['No Finding'].values
    else:
        y = np.stack(df['multi_category_labels'].values)

    if data_augmentation:
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(read_image_data_augmentation).batch(batch_size=batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(read_image).batch(batch_size=batch_size)
    return dataset


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
    path_to_figures: str = args.path_to_figures

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


    label_classes: List[str] = ['No Finding', 'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
                                'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly',
                                'Nodule', 'Mass', 'Hernia']

    training_dataset: tf.data.Dataset = load_dataset(path_to_images=path_to_training_images,
                                                     path_to_pkl=path_to_training_pkl,
                                                     model_type=model_type,
                                                     batch_size=batch_size,
                                                     data_augmentation=augment)
    validation_dataset: tf.data.Dataset = load_dataset(path_to_images=path_to_validation_images,
                                                       path_to_pkl=path_to_validation_pkl,
                                                       model_type=model_type,
                                                       batch_size=batch_size)

    number_of_output_classes: int = 1 if model_type == 'Binary' else 15

    loss: tf.keras.losses
    metric: List[tf.keras.metrics]
    if model_type == 'Binary':
        loss = tf.keras.losses.BinaryCrossentropy()
        metric = ['accuracy']
    else:
        loss = tf.keras.losses.CategoricalCrossentropy()
        metric = [tf.keras.metrics.AUC()]

    callbacks = []
    if lr_scheduler:
        callback_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        callbacks.append(callback_scheduler)

    monitor: str = 'val_accuracy' if model_type == 'Binary' else 'val_auc'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(path_to_save_models, 'checkpoints'),
        save_weights_only=True,
        monitor=monitor,
        mode='max',
        save_best_only=True)
    callbacks.append(model_checkpoint_callback)

    nih_classifier: NiHClassifier = NiHClassifier(number_of_output_classes=number_of_output_classes)
    nih_classifier.compile(optimizer=tf.keras.optimizers.Adam(lr),
                           loss=loss,
                           metrics=metric)

    training_history = nih_classifier.fit(training_dataset,
                                          epochs=initial_epochs,
                                          validation_data=validation_dataset,
                                          callbacks=callbacks)

    datetime_str: str = datetime.now().strftime('%Y%H%M%S')
    plot_loss_acc(training_history, os.path.join(path_to_figures, f'{datetime_str}_model_{model_type}_initial.png'), model_type=model_type)
    nih_classifier.save_weights(os.path.join(path_to_save_models, f'{datetime_str}_model_{model_type}_initial', f'model'))

    nih_classifier.unfreeze_top_layers(fine_tune_top_n=fine_tune_at)

    nih_classifier.compile(optimizer=tf.keras.optimizers.Adam(lr/10),
                           loss=loss,
                           metrics=metric)

    total_epochs = initial_epochs + fine_tuning_epochs

    fine_tuning_history = nih_classifier.fit(training_dataset,
                                             epochs=total_epochs,
                                             initial_epoch=training_history.epoch[-1],
                                             validation_data=validation_dataset,
                                             callbacks=callbacks)

    datetime_str: str = datetime.now().strftime('%Y%H%M%S')
    plot_loss_acc(fine_tuning_history, os.path.join(path_to_figures, f'{datetime_str}_model_{model_type}_{fine_tune_at}_finetuned.png'), model_type=model_type)
    nih_classifier.save_weights(os.path.join(path_to_save_models, f'{datetime_str}_model_{model_type}_{fine_tune_at}_finetuned', 'model'))


if __name__ == '__main__':
    main()