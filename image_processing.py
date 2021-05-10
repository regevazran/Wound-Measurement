import cv2
import numpy as np

class image_process_algo_master:
    def __init__(self, dataset):
        self.dataset = dataset
        self.picture_list = []

    def get_frames_from_dataset(self):
        self.picture_list = self.dataset.get_pic_with_tag(mouse_name=self.dataset.data["Mouse"][0], day=0).pictures

    def preprocess_pic(self, pic):
        pic = np.array(pic)
        pic = cv2.resize(pic, (int(pic.shape[1]*0.4), int(pic.shape[0]*0.4)))
        gray_pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        _, thresh_pic = cv2.threshold(gray_pic, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pic, contours, -1, (0, 0, 0), thickness=3)
        cv2.imshow("try", pic)
        cv2.waitKey(0)

    def get_wound_size(self, preprocess_pic):
        pass

    def start(self):
        self.get_frames_from_dataset()
        for pic in self.picture_list:
            preprocess_pic = self.preprocess_pic(pic)
            # wound_size = self.get_wound_size(preprocess_pic)
            # print(wound_size)
