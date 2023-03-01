import cv2
import numpy as np
import threading


class CamThread(threading.Thread):
    def __init__(self, cam_id):
        super(CamThread, self).__init__()

        self.cam_id = cam_id
        self.frame = np.zeros([512, 512], 'uint8')
        self.event = threading.Event()
        self.daemon = True

    def run(self) -> None:
        cam = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)

        while not self.event.is_set():
            ret, frame = cam.read()

            if ret:
                self.frame = np.copy(frame)

        if cam.isOpened():
            cam.release()

    def get_frame(self):
        return self.frame


































































