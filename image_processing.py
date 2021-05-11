import cv2
import numpy as np
from scipy.spatial import distance

class image_process_algo_master:
    def __init__(self, dataset):
        self.dataset = dataset
        self.picture_list = []

    def get_frames_from_dataset(self):
        self.picture_list = self.dataset.get_pic_with_tag(mouse_name=self.dataset.data["Mouse"][0], day=0).pictures

    def find_center_contour(self, contours, pic):
        # take only the mouse contour
        contours.sort(reverse= True,key = cv2.contourArea)
        biggest_three = []

        # find center of image and draw it (blue circle)
        image_center = np.asarray(pic.shape) / 2
        image_center = tuple(image_center.astype('int32'))
        for i in range(0,2):
            if (i <= len(contours)):
                # find center of each contour
                C = cv2.moments(contours[i])
                center_X = int(C["m10"] / C["m00"])
                center_Y = int(C["m01"] / C["m00"])
                contour_center = (center_X, center_Y)

                # calculate distance to image_center
                distances_to_center = (distance.euclidean(image_center[0:2], contour_center))

                # save to a list of dictionaries
                biggest_three.append({'contour': contours[i], 'distance_to_center': distances_to_center})
        # sort the contours
        sorted_biggest_three = sorted(biggest_three, key=lambda i: i['distance_to_center'])
        return sorted_biggest_three[0]['contour']

    def preprocess_pic(self, pic):
        pic = np.array(pic)
        pic = cv2.resize(pic, (int(pic.shape[1]*0.4), int(pic.shape[0]*0.4)))
        gray_pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        _, thresh_pic = cv2.threshold(gray_pic, 80, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((30, 30), np.uint8)
        closing = thresh_pic
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("thresh", closing)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center_contour = []
        center_contour.append(self.find_center_contour(contours= contours, pic= pic))
        cv2.drawContours(pic, center_contour, -1, (0, 255, 0), thickness=3)
        cv2.imshow("try", pic)
        cv2.waitKey(0)

    def get_wound_size(self, preprocess_pic):
        pass

    def start(self):
        self.get_frames_from_dataset()
        for pic in self.picture_list:
            preprocess_pic = self.preprocess_pic(pic)
        #     # wound_size = self.get_wound_size(preprocess_pic)
        #     # print(wound_size)
       # preprocess_pic = self.preprocess_pic(self.picture_list[0])
