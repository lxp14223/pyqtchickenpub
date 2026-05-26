import cv2


class UsbCamera:
    def __init__(self):
        self.cap = None
        self.camera_index = None

    def open(self, camera_index=0):
        self.release()
        self.camera_index = int(camera_index)

        backends = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        backends.append(cv2.CAP_ANY)

        for backend in backends:
            cap = cv2.VideoCapture(self.camera_index, backend)
            if cap is not None and cap.isOpened():
                self.cap = cap
                return True
            if cap is not None:
                cap.release()

        self.cap = None
        return False

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if not self.is_opened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
