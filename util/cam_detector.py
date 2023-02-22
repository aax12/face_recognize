import numpy as np
import cv2
from threading import Thread
from queue import Queue
from util.misc import resize_image, get_batch_bboxes, change_wh, change_xy, nms


class FaceDetector:
    def __init__(self, detect_config, detect_model, recognize_model, face_vector, face_size, frame_size):
        self.anchor = np.array(detect_config['anchors'][::-1])
        self.scales = np.array(detect_config['anchor_scale'][::-1])
        self.input_size = detect_config['input_size'][:2]

        self.d_model = detect_model
        self.r_model = recognize_model
        self.face_vector = face_vector
        self.face_size = face_size
        self.frame_size = frame_size

        img_w, img_h = frame_size
        in_h, in_w = self.input_size

        self.ratio = in_h / img_h if img_h > img_w else in_w / img_w
        self.x_off = np.round((self.ratio * img_w) / 2. if img_h > img_w else 0.).astype('int32')
        self.y_off = np.round(0. if img_h > img_w else (self.ratio * img_h) / 2.).astype('int32')

        self.frame_image = Queue(8)
        self.bboxes = np.zeros([0, ], 'int32')

        self.thread_flag = True
        self.thread_func = Thread(target=self._face_detect)
        self.thread_func.daemon = True
        self.thread_func.start()

    def __del__(self):
        self.thread_flag = False

    def copy_frame(self, frame):
        if not self.frame_image.full():
            self.frame_image.put(np.copy(frame))

    def get_bboxes(self):
        return np.copy(self.bboxes)

    def get_distance(self):
        pass

    def _face_detect(self):
        while self.thread_flag:
            if not self.frame_image.full():
                continue

            frame_images = []
            while not self.frame_image.empty():
                image = resize_image(self.frame_image.get(), self.input_size)
                frame_images.append(image)

            y = self.d_model.predict_on_batch(np.stack(frame_images))

            valid_bboxes = []
            for bboxes in get_batch_bboxes(self.anchor, self.scales, y):
                if len(bboxes) > 0:
                    bboxes = (bboxes - (self.x_off, self.y_off, self.x_off, self.y_off)) / self.ratio
                    bboxes = change_wh(bboxes)

                    bboxes[..., 2] += bboxes[..., 2] * 0.2
                    bboxes[..., 3] += bboxes[..., 3] * 0.2

                    wh_ratio = bboxes[..., 2] / bboxes[..., 3]
                    mask = np.logical_or(wh_ratio > 1.2, 1 / wh_ratio > 1.2)
                    bboxes = change_xy(bboxes[mask])

                    bboxes[..., 0] = np.maximum(bboxes[..., 0].astype('int32'), 0)
                    bboxes[..., 1] = np.maximum(bboxes[..., 1].astype('int32'), 0)
                    bboxes[..., 2] = np.minimum(bboxes[..., 2].astype('int32'), self.frame_size[0])
                    bboxes[..., 3] = np.minimum(bboxes[..., 3].astype('int32'), self.frame_size[1])

                    valid_bboxes.extend(bboxes)
            if len(valid_bboxes) > 0:
                valid_bboxes = np.array(valid_bboxes)
                _, self.bboxes = nms(valid_bboxes[..., 0], valid_bboxes, 0.3)

    def release(self):
        self.thread_flag = False























































