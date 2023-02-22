from tensorflow.keras.layers import Input, Layer, Lambda, Dense
from tensorflow.keras.layers import TimeDistributed, LSTM, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class WeightedSum(Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        batch, _, vector = input_shape[0]

        return batch, vector

    def call(self, inputs, *args, **kwargs):
        '''
        :param inputs: [batch, class, 256], [batch, class, 1]
        :param args:
        :param kwargs:
        :return: [batch, 512]
        '''

        h_state = inputs[0]
        decision = inputs[1]

        return K.sum(h_state * decision, axis=1)


class L2_distance(Layer):
    def __init__(self, **kwargs):
        super(L2_distance, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        batch, _ = input_shape[0]

        return batch, 1

    def call(self, inputs, *args, **kwargs):
        x1 = inputs[0]
        x2 = inputs[1]

        return K.sqrt(K.sum(K.square(x1 - x2), -1, True))


def face_net(backbone: Model, input_size):
    x_input = Input(input_size)
    x = TimeDistributed(backbone, name='backbone')(x_input)
    x = TimeDistributed(Dense(512, 'sigmoid', name='fc_1'))(x)

    hs = LSTM(256, return_sequences=True)(x)

    x = TimeDistributed(Dense(1, name='fc_2'))(hs)
    x = Softmax(axis=1)(x)
    x = WeightedSum()([hs, x])
    x = Dense(128, use_bias=False, name='fc_3')(x)
    x = Lambda(lambda tensor: K.l2_normalize(tensor, -1))(x)

    return Model(x_input, x)

























