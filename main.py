import cv2
import numpy as np

from sample_data.celeb_a import CelebAParser, MultipleCelebA, ContrastCelebA
from model.face_net import face_net, L2_distance
from model.mobile_net_v1 import mobilenet_v1
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from loss_function.loss import hard_triplet_loss, OnlineTripletLoss
import tensorflow as tf
from tensorflow.keras.utils import Progbar


def main():
    train_celeba = CelebAParser('E:/data_set/CelebA/img/face_images/train_224.record', 64)
    train_sample = MultipleCelebA(train_celeba, 16, 4, 3)

    valid_celeba = CelebAParser('E:/data_set/CelebA/img/face_images/valid_224.record', 64)
    valid_sample = MultipleCelebA(valid_celeba, 16, 4, 3)

    input_size = (None, 160, 160, 3)
    backbone = mobilenet_v1(input_size[1:])
    backbone.load_weights('weights/pre_trained/mobile_v1.h5', True)
    model = face_net(backbone, input_size)

    for layer in model.layers:
        if layer.name == 'backbone':
            layer.trainable = False

    model.compile(
        optimizer=Adam(1e-3),
        loss=OnlineTripletLoss(0.2, hard_triplet_loss)
    )

    model.fit(
        x=train_sample,
        epochs=1,
        steps_per_epoch=100,
        verbose=1,
        validation_steps=30,
        validation_data=valid_sample
    )

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(1e-4),
        loss=OnlineTripletLoss(0.2, hard_triplet_loss)
    )

    check_point = ModelCheckpoint(
        filepath='weights/face_net_160_3.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    )
    stopper = EarlyStopping(
        monitor='val_loss',
        verbose=1,
        patience=20
    )
    reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        cooldown=3,
        verbose=1,
        min_lr=1e-5
    )
    terminate_nan = TerminateOnNaN()

    model.fit(
        x=train_sample,
        epochs=100,
        steps_per_epoch=100,
        verbose=1,
        callbacks=[check_point, stopper, reducer, terminate_nan],
        validation_steps=30,
        validation_data=valid_sample
    )


def insert_images(images):
    amounts, height, width, channels = images.shape
    wh_grid = np.ceil(np.sqrt(amounts)).astype('int32')
    canvus = np.zeros([height * wh_grid, width * wh_grid, channels], 'uint8')

    for n, image in enumerate(images):
        x = n % wh_grid
        y = n // wh_grid

        canvus[y * height:(y + 1) * height, x * width:(x + 1) * width] = image

    return canvus


def main2():
    input_size = [None, 160, 160, 3]
    backbone = mobilenet_v1(input_size[1:])
    face_model = face_net(backbone, input_size)
    face_model.load_weights('weights/face_net_160_3.h5')

    x1_input = Input(input_size)
    x2_input = Input(input_size)

    x1 = face_model(x1_input)
    x2 = face_model(x2_input)

    y_output = L2_distance()([x1, x2])

    model = Model([x1_input, x2_input], y_output)

    celeba = CelebAParser('E:/data_set/CelebA/img/face_images/test_224.record', 64)
    sample = ContrastCelebA(celeba, 32, 3)

    bar = Progbar(100)
    count = []
    for i in range(100):
        x, y = next(sample)
        my = model.predict_on_batch(x)

        logic = np.where(np.squeeze(my, -1) < 0.95, 1, 0)
        count.extend(y * logic + (1 - y) * (1 - logic))
        bar.update(i + 1)

    print(np.mean(count))


def main3():
    celeba = CelebAParser('E:/data_set/CelebA/img/face_images/test_224.record', 64)
    sample = ContrastCelebA(celeba, 32, 4)

    input_size = [None, 160, 160, 3]
    backbone = mobilenet_v1(input_size[1:])
    face_model = face_net(backbone, input_size)
    face_model.load_weights('weights/face_net_160_3.h5')

    x1_input = Input(input_size)
    x2_input = Input(input_size)

    x1 = face_model(x1_input)
    x2 = face_model(x2_input)

    y_output = L2_distance()([x1, x2])

    model = Model([x1_input, x2_input], y_output)

    cv2.namedWindow('x1')
    cv2.namedWindow('x2')

    key = 0
    for i in range(100):
        x, t = next(sample)
        y = model.predict_on_batch(x)

        for x1, x2, y_pred, truth in zip(x[0], x[1], y, t):
            x1 = insert_images(x1)
            x2 = insert_images(x2)

            print(y_pred, 1 - truth)

            cv2.imshow('x1', x1)
            cv2.imshow('x2', x2)

            key = cv2.waitKey()

            if key == 27:
                break
        if key == 27:
            break


if __name__ == '__main__':
    main3()








































