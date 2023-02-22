from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, ReLU
from tensorflow.keras.layers import UpSampling2D, Concatenate, MaxPooling2D, LeakyReLU
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


def ConvBN_1(n_filters, filter_size, strides, layer_name, **kwargs):
    def wrapper(x):
        kwargs_ = kwargs.copy()
        act_func = kwargs_.pop('act_func')
        alpha = kwargs_.pop('alpha')

        x = Conv2D(n_filters, filter_size, strides, name=layer_name, **kwargs_)(x)
        x = BN(name=layer_name + '_bn')(x)
        x = act_func(alpha, name=layer_name + '_relu')(x)

        return x
    return wrapper


def ConvBN_2(n_filters, filter_size, strides, kwargs):
    def wrapper(x):
        kwargs_ = kwargs.copy()

        x = Conv2D(n_filters, filter_size, strides, **kwargs_)(x)
        x = BN()(x)
        x = ReLU(max_value=6.)(x)

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
            x = ReLU(max_value=6., name=act_name)(x)

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
        x = ReLU(max_value=6., name=act_name)(x)

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


def ConvBlock(n_out_channels, strides, n_layer, **kwargs):
    def wrapper(x):
        kwargs_ = kwargs.copy()
        act_func = kwargs_.pop('act_func')
        alpha = kwargs_.pop('alpha')

        x = DepthwiseConv2D((3, 3), strides, name=f'conv_dw_{n_layer}', **kwargs_)(x)
        x = BN(name=f'conv_dw_{n_layer}_bn')(x)
        x = act_func(alpha, name=f'conv_dw_{n_layer}_relu')(x)
        x = ConvBN_1(n_out_channels, (1, 1), (1, 1), f'conv_pw_{n_layer}', **kwargs)(x)

        return x
    return wrapper


def TinyFaceDetection(input_size):
    param = {
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(5e-4),
        'use_bias': False,
        'padding': 'same',
        'act_func': LeakyReLU,
        'alpha': 0.1
    }

    def ActFunc(y_pred):
        shape = K.shape(y_pred)
        y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 3, 4 + 1))

        xy = K.sigmoid(y_pred[..., :2])
        wh = y_pred[..., 2:4]
        conf = K.sigmoid(y_pred[..., 4:5])

        return K.concatenate([xy, wh, conf])

    x_input = Input(input_size)
    x = Lambda(lambda tensor: tensor / 255.)(x_input)

    x = ConvBN_1(16, (3, 3), (1, 1), 'conv_1', **param)(x)
    x = MaxPooling2D((2, 2), (2, 2), 'same')(x)
    x = ConvBN_1(32, (3, 3), (1, 1), 'conv_2', **param)(x)
    x = MaxPooling2D((2, 2), (2, 2), 'same')(x)
    x = ConvBN_1(64, (3, 3), (1, 1), 'conv_3', **param)(x)
    x = MaxPooling2D((2, 2), (2, 2), 'same')(x)
    x = ConvBN_1(128, (3, 3), (1, 1), 'conv_4', **param)(x)
    x = MaxPooling2D((2, 2), (2, 2), 'same')(x)
    branch1 = ConvBN_1(256, (3, 3), (1, 1), 'conv_5', **param)(x)
    x = MaxPooling2D((2, 2), (2, 2), 'same')(branch1)
    x = ConvBN_1(512, (3, 3), (1, 1), 'conv_6', **param)(x)
    x = MaxPooling2D((2, 2), (1, 1), 'same')(x)
    x = ConvBN_1(1024, (3, 3), (1, 1), 'conv_7', **param)(x)
    branch2 = ConvBN_1(256, (1, 1), (1, 1), 'conv_8', **param)(x)

    x = ConvBN_1(512, (3, 3), (1, 1), 'conv_9', **param)(branch2)
    x = Conv2D(3 * (4 + 1), (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)  # 15 [batch, 13, 13, 4 + 1] x32
    y1 = Lambda(ActFunc, name='y1')(x)

    x = ConvBN_1(128, (1, 1), (1, 1), 'conv_10', **param)(branch2)
    x = UpSampling2D()(x)
    x = Concatenate()([x, branch1])
    x = ConvBN_1(256, (3, 3), (1, 1), 'conv_11', **param)(x)
    x = Conv2D(3 * (4 + 1), (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)  # [batch, 26, 26, 4 + 1] x16
    y2 = Lambda(ActFunc, name='y2')(x)

    model = Model(x_input, [y1, y2])

    '''
    if doFreeze:
        for i in range(37):
            model.layers[i].trainable = False
    '''
    return model

