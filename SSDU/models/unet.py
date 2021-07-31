import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, AvgPool2D, Conv2DTranspose, Concatenate
import numpy as np


def conv_block(input_tensor, out_chans, drop_prob):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    out = Conv2D(out_chans, kernel_size=3, padding="same", use_bias=False)(input_tensor)
    # out = InstanceNormalization(?)(out)
    out = LeakyReLU(0.2)(out)
    out = Dropout(drop_prob)(out)
    out = Conv2D(out_chans, kernel_size=3, padding="same", use_bias=False)(out)
    # out = InstanceNormalization(?)(out)
    out = LeakyReLU(0.2)(out)
    out = Dropout(drop_prob)(out)

    return out


def transpose_conv_block(input_tensor, out_chans):

    out = Conv2DTranspose(out_chans, kernel_size=2, strides=(2, 2), use_bias=False)(input_tensor)
    # Instance Norm
    out = LeakyReLU(0.2)(out)

    return out


def unet(input_tensor, out_chans, chans, num_pooling_layers, drop_prob):
    stack = []
    output = input_tensor

    ch = chans
    # apply down-sampling layers
    for _ in range(num_pooling_layers):
        output = conv_block(output, ch, drop_prob)
        ch *= 2
        stack.append(output)
        output = AvgPool2D((2, 2), strides=(2, 2), padding="valid")(output)

    output = conv_block(output, ch, drop_prob)

    ch //= 2
    # apply up-sampling layers
    for _ in range(num_pooling_layers - 1):
        downsample_out = stack.pop()
        output = transpose_conv_block(output, ch)
        output = Concatenate(axis=-1)([output, downsample_out])
        output = conv_block(output, ch, drop_prob)
        ch //= 2

    output = transpose_conv_block(output, ch)
    output = conv_block(output, ch, drop_prob)
    output = Conv2D(out_chans, kernel_size=1, strides=(1, 1))(output)

    return output
