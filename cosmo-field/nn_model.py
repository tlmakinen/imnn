import numpy as np
import tensorflow as tf

def field2Dmodel(input_shape, n_summaries, strides=2, kernel=3,
                growth_factor=2, filters=8, dense_size=1):
    s = strides
    k = kernel
    growth_factor = growth_factor
    filters = filters
    dense_size = dense_size

    size = input_shape[-1]

    layers = [tf.keras.Input(shape=input_shape),
         tf.keras.layers.Reshape(input_shape[::-1])]
    while size > dense_size:
        # update filters
        filters *= growth_factor
        layers += [tf.keras.layers.Conv2D(filters,k, strides=s, padding='same'),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.LeakyReLU(0.2)]
        # update tensor size
        size //= growth_factor

    # add final Conv2D layer to final output
    layers += [
     tf.keras.layers.Conv2D(SN.n_summaries, kernel_size=(1, 1), strides=(1, 1), padding="valid"),
     tf.keras.layers.Flatten()
    ]

    return tf.keras.Sequential(layers)
