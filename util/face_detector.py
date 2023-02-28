import cv2
import numpy as np
import copy
import time

from threading import Thread
from tensorflow.keras.models import Model

from util.misc import IOU, get_bboxes, change_wh, change_xy


class Detector:
    def __init__(
            self,
            config: dict,
            detector_model: Model,
            recognize_model: Model,
            face_vector,
            get_frames,
            frame_size,
            face_size
    ):
        self.frame_size = frame_size
        self.input_size = config['input_size']
        self.scales = config['anchor_scale'][::-1]
        self.anchors = config['anchors'][::-1]
        self.detector_model = detector_model
        self.recognize_model = recognize_model
        self.face_vector = np.array(face_vector)
        self.get_frames = get_frames
        self.face_size = face_size

        resize_func, offset, ratio = self._resize_image(frame_size[::-1], self.input_size)
        self.resize = resize_func
        self.offset = offset
        self.ratio = ratio

        self.frame = np.zeros(self.input_size, 'uint8')

        self.detected_faces = [[]]
        self.prev_bboxes = np.zeros([1, 4], 'int32')
        self.distances = []

        self.thread_flag = True
        self.is_copy = False

        self.thread = Thread(target=self._detecting_face)
        self.thread.daemon = True
        self.thread.start()

    def __del__(self):
        self.thread_flag = False

    def _detecting_face(self):
        while self.thread_flag:
            if self.is_copy:
                continue

            frame = self.frame
            x_input = self.resize(frame)
            y = self.detector_model.predict_on_batch(np.expand_dims(x_input, 0))
            bboxes = get_bboxes(self.anchors, self.scales, y)

            if len(bboxes) > 0:
                bboxes_xy = (bboxes - self.offset) * self.ratio
                bboxes_wh = change_wh(bboxes_xy)

                bboxes_wh[..., 2] += bboxes_wh[..., 2] * 0.2
                bboxes_wh[..., 3] += bboxes_wh[..., 3] * 0.2

                bboxes_xy = change_xy(bboxes_wh)

                bboxes_xy[..., 0] = np.maximum(bboxes_xy[..., 0], 0)
                bboxes_xy[..., 1] = np.maximum(bboxes_xy[..., 1], 0)
                bboxes_xy[..., 2] = np.minimum(bboxes_xy[..., 2], self.frame_size[0])
                bboxes_xy[..., 3] = np.minimum(bboxes_xy[..., 3], self.frame_size[1])

                wh_ratio = (bboxes_xy[:, 2] - bboxes_xy[:, 0]) / (bboxes_xy[:, 3] - bboxes_xy[:, 1])
                mask = np.logical_or(wh_ratio < 1.2, 1 / wh_ratio < 1.2)

                self._get_faces(frame, bboxes_xy[mask])

            time.sleep(0.03)

            self._recognizing_face()

    def _recognizing_face(self):
        tmp_distance = []

        for group in self.detected_faces:
            if len(group) == self.get_frames:
                x_image = np.expand_dims(np.stack(group), 0)
                distance = self.recognize_model.predict_on_batch([x_image, self.face_vector])
                tmp_distance.extend(np.squeeze(distance, 0))

        if len(tmp_distance) > 0:
            self.distances = tmp_distance

    def _get_faces(self, image, bboxes):
        if len(bboxes) == 0:
            return

        iou = IOU(self.prev_bboxes, bboxes)
        bboxes = np.round(bboxes).astype('int32')

        max_index = np.argmax(iou, -1)
        y_indices = np.arange(len(self.prev_bboxes))
        best_mask = iou[y_indices, max_index] > 0.6

        selected_x_indices = max_index[best_mask]
        selected_y_indices = np.arange(len(self.prev_bboxes))[best_mask]

        for x_index, y_index in zip(selected_x_indices, selected_y_indices):
            if len(self.detected_faces[y_index]) >= self.get_frames:
                del self.detected_faces[y_index][0]
            img = self._resize_face(image, bboxes[x_index])
            self.detected_faces[y_index].append(img)

        not_select_x = [n for n in range(len(bboxes)) if n not in selected_x_indices]

        tmp = [self.detected_faces[n] for n in selected_y_indices]
        self.detected_faces = tmp

        for index in not_select_x:
            img = self._resize_face(image, bboxes[index])
            self.detected_faces.append([img, ])

        self.prev_bboxes = bboxes[selected_x_indices.tolist() + not_select_x]

    def _resize_face(self, image, bbox):
        x1, y1, x2, y2 = bbox
        return cv2.resize(image[y1:y2, x1:x2], self.face_size)

    def get_data(self):
        self.is_copy = True
        faces = copy.deepcopy(self.detected_faces)
        bboxes = self.prev_bboxes.copy()
        distance = self.distances.copy()
        self.is_copy = False

        return faces, bboxes, distance

    def copy_frame(self, frame):
        self.is_copy = True
        self.frame = np.copy(frame)
        self.is_copy = False

    @staticmethod
    def _resize_image(frame_size, output_size):
        img_h, img_w = frame_size[:2]
        out_h, out_w = output_size[:2]

        if img_w < img_h:
            ratio = out_h / img_h
            width = np.round(ratio * img_w).astype('int32')
            x_start = (out_w - width) // 2

            def wrapper(image):
                canvas = np.zeros([out_h, out_w, 3], 'uint8')
                canvas[:, x_start:x_start + width] = cv2.resize(image, [width, out_h])
                return canvas
            return wrapper, (x_start, 0, x_start, 0), 1 / ratio
        else:
            ratio = out_w / img_w
            height = np.round(ratio * img_h).astype('int32')
            y_start = (out_h - height) // 2

            def wrapper(image):
                canvas = np.zeros([out_h, out_w, 3], 'uint8')
                canvas[y_start:y_start + height] = cv2.resize(image, [out_w, height])
                return canvas
            return wrapper, (0, y_start, 0, y_start), 1 / ratio
