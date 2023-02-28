import cv2
import json
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from model.face_detecter import TinyFaceDetection
from model.mobile_net_v1 import mobilenet_v1
from model.face_net import face_net, L2_distance
from util.face_detector import Detector


def main():
    face_size = [None, 160, 160, 3]

    with open('yolov3_tiny_cfg.json', 'r') as fp:
        config = json.load(fp)

    with open('my_face_vector.json', 'r') as fp:
        face_vector = json.load(fp)['vector']

    detector_model = TinyFaceDetection(config['input_size'])
    detector_model.load_weights('weights/pre_trained/face_detect_2stage.h5')
    detector_model.trainable = False

    backbone = mobilenet_v1(face_size[1:])
    recognize = face_net(backbone, face_size)
    recognize.load_weights('weights/face_net_160_3.h5')

    x_input = Input(face_size)
    v_input = Input([128, ])

    x = recognize(x_input)
    y_output = L2_distance()([x, v_input])

    recognize_model = Model([x_input, v_input], y_output)
    recognize_model.trainable = False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = Detector(
        config,
        detector_model,
        recognize_model,
        face_vector,
        4,
        (frame_w, frame_h),
        face_size[1:3]
    )

    cv2.namedWindow('bboxes')

    while 27 != cv2.waitKey(1):
        ret, frame = cap.read()
        if ret:
            detector.copy_frame(frame)

            _, bboxes, distances = detector.get_data()
            for bbox, distance in zip(bboxes, distances):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255))
                cv2.putText(
                    frame,
                    f'{distance:3f}',
                    (x1 + 2, y1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0) if distance < 0.9 else (0, 0, 255)
                )
            cv2.imshow('bboxes', frame)

    if cap.isOpened():
        cap.release()


if __name__ == '__main__':
    main()




























































