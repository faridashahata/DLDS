import tensorflow as tf

from typing import *


class NiHClassifier(tf.keras.Model):
    def __init__(self, number_of_output_classes: int,
                 image_shape: Tuple[int, int, int] = (224, 224, 3)):
        super().__init__()
        self.number_of_output_classes: int = number_of_output_classes
        self.image_shape: Tuple[int, int, int] = image_shape

        self.pretrained_resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                                           weights="imagenet",
                                                                           input_shape=image_shape)
        self.pretrained_resnet50.trainable = False

        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.prediction_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.number_of_output_classes, activation=tf.keras.activations.sigmoid)
        ])

        self.build(input_shape=(None, image_shape[0], image_shape[1], image_shape[2]))

    def unfreeze_top_layers(self, fine_tune_top_n: int):
        self.pretrained_resnet50.trainable = True

        number_of_layers: int = len(self.pretrained_resnet50.layers)
        layers_to_freeze: int = number_of_layers - fine_tune_top_n

        for i in range(layers_to_freeze):
            self.pretrained_resnet50.layers[i].trainable = False

    def call(self, inputs, training=None, mask=None):
        resnet_features = self.pretrained_resnet50(inputs, training=training)
        avg_pooling_features = self.global_average_pooling(resnet_features)
        predictions = self.prediction_layer(avg_pooling_features)
        return predictions
