import cv2
import json
import os
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from model.face_detecter import TinyFaceDetection
from model.mobile_net_v1 import mobilenet_v1
from model.face_net import face_net, L2_distance
from util.misc import get_bboxes, IOU
from util.cam_detector import FaceDetector
from tensorflow.keras.utils import Progbar


def read_partition(path, mode):
    with open(path, 'rt') as f:
        text = f.readlines()

    mask = np.zeros(len(text), 'bool')
    n_mode = {'train': '0', 'valid': '1', 'test': '2'}[mode]

    for i, line in enumerate(text):
        _, part = line.split()

        if part == n_mode:
            mask[i] = True

    return mask


def read_identity(path, mask):
    with open(path, 'rt') as f:
        text = f.readlines()

    names = []
    ident = []
    for line in text:
        name, n = line.split()

        ident.append(int(n))
        names.append(name)

    ident = np.array(ident)[mask]
    indices = np.arange(len(text))[mask]

    groups = []
    for n in set(ident):
        groups.append(indices[ident == n].tolist())

    return names, groups


def padding(images, image_size):
    canvus = np.ones((len(images), *image_size), 'uint8')
    canvus_h, canvus_w = image_size[:2]

    for i, img in enumerate(images):
        img_h, img_w = img.shape[:2]
        x1 = int((canvus_w - img_w) * 0.5)
        y1 = int((canvus_h - img_h) * 0.5)

        canvus[i, y1:y1+img_h, x1:x1+img_w] = img

    return canvus


def valid_bbox(bboxes):
    val_bbox = []

    for bbox in bboxes:
        if len(bbox) > 0:
            x1 = bbox[..., 0]
            y1 = bbox[..., 1]
            x2 = bbox[..., 2]
            y2 = bbox[..., 3]

            area = (x2 - x1) * (y2 * y1)
            max_index = np.argmax(area)
            val_bbox.append(bbox[max_index])
        else:
            val_bbox.append([])

    return val_bbox


def draw_rect(image, bbox):
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255))


def expand_images(images, bboxes, extend, resize):
    height, width = resize[:2]
    face_images = np.zeros((len(images), *resize), 'uint8')

    for i, (image, bbox) in enumerate(zip(images, bboxes)):
        x1, y1, x2, y2 = bbox
        x1 -= extend
        y1 -= extend
        x2 += extend
        y2 += extend

        face_images[i] = cv2.resize(image[y1:y2, x1:x2], (width, height))

    return face_images


def main():
    face_size = (None, 160, 160, 3)

    with open('yolov3_tiny_cfg.json', 'r') as fp:
        config = json.load(fp)

    model = TinyFaceDetection(config['input_size'])
    model.load_weights('weights/pre_trained/face_detect_2stage.h5')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = FaceDetector(config, model, None, None, (160, 160, 3), (frame_h, frame_w, 3))

    cv2.namedWindow('bboxes')

    while 27 != cv2.waitKey(1):
        ret, frame = cap.read()
        if ret:
            detector.copy_frame(frame)

            for bbox in detector.get_bboxes():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255))
            cv2.imshow('bboxes', frame)

    if cap.isOpened():
        cap.release()


def main2():
    face_size = (None, 160, 160, 3)

    backbone = mobilenet_v1(face_size[1:])
    face_model = face_net(backbone, face_size)
    face_model.load_weights('weights/face_net_160_3.h5', True)

    directory = 'E:/data_set'
    files = ['my_face_1', 'my_face_2', 'my_face_3', 'my_face_fail_2']
    faces = []

    for file_name in files:
        image = cv2.imread(os.path.join(directory, file_name) + '.jpg')
        image = cv2.resize(image, (160, 160))
        faces.append(image)

    faces = np.stack(faces)

    cv2.namedWindow('face')
    for image in faces:
        cv2.imshow('face', image)
        if cv2.waitKey() == 27:
            break
    cv2.destroyWindow('face')

    y = face_model.predict_on_batch(np.expand_dims(faces, 0))
    face_vector = {'vector': y.tolist()}

    with open('my_face_vector.json', 'w') as fp:
        json.dump(face_vector, fp, indent=4)


if __name__ == '__main__':
    main()
































































