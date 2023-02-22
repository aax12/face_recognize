from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, LeakyReLU, Dense
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAvgPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def ConvBN(n_filters, filter_size, strides, kwargs):
    def wrapper(x):
        kwargs_ = kwargs.copy()

        x = Conv2D(n_filters, filter_size, strides, **kwargs_)(x)
        x = BN()(x)
        x = LeakyReLU(alpha=0.1)(x)

        return x

    return wrapper


def ExpandConvBN(in_ch, ex_factor, kwargs):
    def wrapper(x):
        kwargs_ = kwargs.copy()
        block_name = kwargs_.pop('block_name')

        conv_name = block_name + '_expand'
        bn_name = block_name + '_expand_BN'
        act_name = block_name + '_expand_relu'

        if ex_factor > 1:
            x = Conv2D(in_ch * ex_factor, [1, 1], [1, 1], name=conv_name, **kwargs_)(x)
            x = BN(name=bn_name)(x)
            x = LeakyReLU(alpha=0.1, name=act_name)(x)

        return x

    return wrapper


def DWConvBN(strides, kwargs):
    def wrapper(x):
        kwargs_ = kwargs.copy()
        block_name = kwargs_.pop('block_name')

        conv_name = block_name + '_depthwise'
        bn_name = block_name + '_depthwise_BN'
        act_name = block_name + '_depthwise_relu'

        x = DepthwiseConv2D([3, 3], strides, name=conv_name, **kwargs_)(x)
        x = BN(name=bn_name)(x)
        x = LeakyReLU(alpha=0.1, name=act_name)(x)

        return x

    return wrapper


def ProjectConvBN(out_ch, kwargs):
    def wrapper(x):
        kwargs_ = kwargs.copy()
        block_name = kwargs_.pop('block_name')

        conv_name = block_name + '_project'
        bn_name = block_name + '_project_BN'

        x = Conv2D(out_ch, [1, 1], [1, 1], name=conv_name, **kwargs_)(x)
        x = BN(name=bn_name)(x)

        return x

    return wrapper


def BottleNeck(in_ch, tcns, kwargs):
    def wrapper(x):
        t, c, n, s = tcns

        kwargs_ = kwargs.copy()
        block_count = kwargs_.pop('block_count')

        for i in range(n):
            kwargs_['block_name'] = f'block_{block_count}'

            if i == 0:
                x = ExpandConvBN(in_ch, t, kwargs_)(x)
                x = DWConvBN((s, s), kwargs_)(x)
                x = ProjectConvBN(c, kwargs_)(x)
            else:
                add_x = x
                x = ExpandConvBN(c, t, kwargs_)(x)
                x = DWConvBN((1, 1), kwargs_)(x)
                x = ProjectConvBN(c, kwargs_)(x)
                x = Add(name=kwargs_['block_name'] + '_add')([x, add_x])

            block_count += 1
        kwargs['block_count'] = block_count

        return x

    return wrapper


def MobileNet_v2(input_size):
    param = {
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(5e-4),
        'use_bias': False,
        'padding': 'same',
        'block_count': 0
    }

    x_input = Input(input_size, name='image_input')
    x = Lambda(lambda tensor: tensor / 255., name='normalization')(x_input)

    kwargs_1 = param.copy()
    kwargs_1.pop('block_count')
    kwargs_1['name'] = 'conv2d_1'

    kwargs_2 = kwargs_1.copy()
    kwargs_2['name'] = 'conv2d_2'

    x = ConvBN(32, [3, 3], [2, 2], kwargs_1)(x)
    x = BottleNeck(32, [1, 16, 1, 1], param)(x)
    x = BottleNeck(16, [6, 24, 2, 2], param)(x)
    x = BottleNeck(24, [6, 32, 3, 2], param)(x)
    x = BottleNeck(32, [6, 64, 4, 2], param)(x)
    x = BottleNeck(64, [6, 96, 3, 1], param)(x)
    x = BottleNeck(96, [6, 160, 3, 2], param)(x)
    x = BottleNeck(160, [6, 320, 1, 1], param)(x)
    x = ConvBN(1280, [1, 1], [1, 1], kwargs_2)(x)
    x = GlobalAvgPool2D()(x)

    return Model(x_input, x, name='backbone')
























