import numpy as np
import cv2
from threading import Thread
from util.misc import get_bboxes, change_wh, change_xy, IOU, resize_image


class FaceDetector:
    def __init__(self, detect_config, detect_model, recognize_model, face_vector, face_size, frame_size):
        self.anchor = np.array(detect_config['anchors'][::-1])
        self.scales = np.array(detect_config['anchor_scale'][::-1])
        self.input_size = detect_config['input_size']

        self.d_model = detect_model
        self.r_model = recognize_model
        self.face_vector = face_vector
        self.face_size = face_size
        self.frame_size = frame_size

        self.resize_func, self.offset, self.ratio = self._resize_image(frame_size, self.input_size)

        self.frame_image = np.zeros(frame_size, 'uint8')
        self.bboxes = np.zeros([1, 4], 'int32')

        self.image_amounts = 12
        self.is_copy = False

        self.thread_flag = True
        self.thread_func = Thread(target=self._face_detect)
        self.thread_func.daemon = True
        self.thread_func.start()

    def __del__(self):
        self.thread_flag = False

    def copy_frame(self, frame):
        self.is_copy = True
        self.frame_image[:] = frame
        self.is_copy = False

    def get_bboxes(self):
        return np.copy(self.bboxes)

    def get_distance(self):
        pass

    def _face_detect(self):
        valid_faces = []

        while self.thread_flag:
            if self.is_copy:
                continue

            face_bboxes = np.array([], 'float32')

            frame_image = self.resize_func(np.stack(self.frame_image))
            y = self.d_model.predict_on_batch(np.expand_dims(frame_image, 0))

            bboxes = get_bboxes(self.anchor, self.scales, y)
            if len(bboxes) > 0:
                bboxes_xy = (bboxes - self.offset) * self.ratio
                bboxes_wh = change_wh(bboxes_xy)

                bboxes_wh[..., 2] += bboxes_wh[..., 2] * 0.2
                bboxes_wh[..., 3] += bboxes_wh[..., 3] * 0.2

                bboxes_xy = change_xy(bboxes_wh)

                bboxes_xy[..., 0] = np.maximum(bboxes_xy[..., 0], 0)
                bboxes_xy[..., 1] = np.maximum(bboxes_xy[..., 1], 0)
                bboxes_xy[..., 2] = np.minimum(bboxes_xy[..., 2], self.frame_size[1])
                bboxes_xy[..., 3] = np.minimum(bboxes_xy[..., 3], self.frame_size[0])

                wh_ratio = (bboxes_xy[:, 2] - bboxes_xy[:, 0]) / (bboxes_xy[:, 3] - bboxes_xy[:, 1])
                mask = np.logical_or(wh_ratio < 1.2, 1 / wh_ratio < 1.2)

                bboxes = bboxes_xy[mask]

                if len(face_bboxes) > 0:
                    iou = IOU(face_bboxes, bboxes)
                    indices = np.argmax(iou, -1)[iou > 0.7 | iou == 0.]
                    face_bboxes

                else:
                    face_bboxes = bboxes
                    for img in self._crop_faces(self.frame_image, bboxes):
                        valid_faces.append([img])


    def _crop_faces(self, image, bboxes):
        faces = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            face = cv2.resize(image[y1:y2, x1:x2], self.face_size)
            faces.append(face)

        return faces

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

    def release(self):
        self.thread_flag = False























































