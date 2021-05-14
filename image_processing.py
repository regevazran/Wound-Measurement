import cv2
import numpy as np
from scipy.spatial import distance

class image_process_algo_master:
    def __init__(self, dataset):
        self.dataset = dataset
        self.picture_list = []

    def get_frames_from_dataset(self,day=0):

        self.picture_list = self.dataset.get_pic_with_tag(mouse_name=self.dataset.data["Mouse"][0], day=day).pictures

    def find_center_contour(self, contours, pic):

        # take only the mouse contour
        contours.sort(reverse= True,key = cv2.contourArea)
        biggest_three = []

        # find center of image and draw it (blue circle)
        image_center = np.asarray(pic.shape) / 2
        image_center = tuple(image_center.astype('int32'))
        for i in range(0,2):
            if (i < len(contours)):
                # find center of each contour
                C = cv2.moments(contours[i])
                if C["m10"] == 0 or C["m00"] == 0 or C["m01"] == 0: continue
                center_Y = int(C["m10"] / C["m00"])
                center_X = int(C["m01"] / C["m00"])
                contour_center = (center_X, center_Y)

                # calculate distance to image_center
                distances_to_center = (distance.euclidean(image_center[0:2], contour_center))

                # save to a list of dictionaries
                biggest_three.append({'contour': contours[i], 'distance_to_center': distances_to_center})
        # sort the contours
        sorted_biggest_three = sorted(biggest_three, key=lambda i: i['distance_to_center'])
        return sorted_biggest_three[0]['contour']
    def get_contours(self, pic):
        _, thresh_pic = cv2.threshold(pic, 80, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((30, 30), np.uint8)
        closing = thresh_pic
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("thresh", closing)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    def cut_mouse(self,pic,contour):
        x, y, w, h = cv2.boundingRect(contour[0])
        pic_copy = pic.copy()
        print("x:",x,"y:",y,"h:",h,"w:",w)
        pic_copy[0:y,:] = (255,255,255)
        pic_copy[:,0:x] = (255,255,255)
        pic_copy[y+h:pic_copy.shape[0],:] = (255,255,255)
        pic_copy[:,x + w: pic_copy.shape[1]] = (255,255,255)
        return pic_copy
    def preprocess_pic(self, pic):
        pic = np.array(pic)
        pic = cv2.resize(pic, (int(pic.shape[1]*0.4), int(pic.shape[0]*0.4)))
        gray_pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

        orig_contours = self.get_contours(gray_pic)
        center_contour = []
        center_contour.append(self.find_center_contour(contours=orig_contours, pic= pic))

        cropped = self.cut_mouse(pic,center_contour)
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        contours = self.get_contours(gray_cropped)
        cropped_center_contour = []
        cropped_center_contour.append(self.find_center_contour(contours= contours, pic= pic))
        cv2.drawContours(cropped, cropped_center_contour, -1, (0, 255, 0), thickness=3)
        cv2.imshow("cropped", cropped)

        cv2.drawContours(pic, center_contour, -1, (0, 255, 0), thickness=3)
        return pic


    def get_wound_size(self, preprocess_pic):
        pass

    def start(self):
        for day in range(0,10):
            self.get_frames_from_dataset(day)
            for pic in self.picture_list:
                preprocess_pic = self.preprocess_pic(pic)
            #     # wound_size = self.get_wound_size(preprocess_pic)
            #     # print(wound_size)
           # preprocess_pic = self.preprocess_pic(self.picture_list[0])
                #cv2.imshow("try", preprocess_pic)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break