from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dense, Lambda
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAvgPool2D
from tensorflow.keras.models import Model


def conv_bn(amounts, filter_size, strides, layer_name):
    def wrapper(x):
        x = Conv2D(amounts, filter_size, strides, padding='same', use_bias=False, name=layer_name)(x)
        x = BN(name=layer_name + '_bn')(x)
        x = LeakyReLU(0.1, name=layer_name + '_relu')(x)

        return x
    return wrapper


def conv_block(amounts, strides, n_layer):
    def wrapper(x):
        x = DepthwiseConv2D([3, 3], strides, 'same', use_bias=False, name=f'conv_dw_{n_layer}')(x)
        x = BN(name=f'conv_dw_{n_layer}_bn')(x)
        x = LeakyReLU(0.1, name=f'conv_dw_{n_layer}_relu')(x)
        x = conv_bn(amounts, (1, 1), (1, 1), f'conv_pw_{n_layer}')(x)

        return x
    return wrapper


def mobilenet_v1(input_size):
    x_input = Input(input_size, name='x_input')
    x = Lambda(lambda tensor: tensor / 255., name='norm_pixel')(x_input)
    x = conv_bn(32, [3, 3], [2, 2], 'conv1')(x)
    x = conv_block(64, [1, 1], 1)(x)
    x = conv_block(128, [2, 2], 2)(x)
    x = conv_block(128, [1, 1], 3)(x)
    x = conv_block(256, [2, 2], 4)(x)
    x = conv_block(256, [1, 1], 5)(x)
    x = conv_block(512, [2, 2], 6)(x)
    x = conv_block(512, [1, 1], 7)(x)
    x = conv_block(512, [1, 1], 8)(x)
    x = conv_block(512, [1, 1], 9)(x)
    x = conv_block(512, [1, 1], 10)(x)
    x = conv_block(512, [1, 1], 11)(x)
    x = conv_block(1024, [2, 2], 12)(x)
    x = conv_block(1024, [1, 1], 13)(x)
    x = GlobalAvgPool2D()(x)

    return Model(x_input, x)



















































