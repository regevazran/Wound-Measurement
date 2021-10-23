import math

import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from yolo.yolo_demo import *
from find_wound_playGround import canny_with_trackbar

import pixellib


class image_process_algo_master:
    def __init__(self, dataset):
        self.dataset = dataset
        self.picture_list = []
        self.cur_frame = None

    # ----------------------not in use------------------------ #
    def get_frames_from_dataset(self, day=0):
        self.picture_list = self.dataset.get_pic_with_tag(mouse_name=self.dataset.data["Mouse"][0], day=day).pictures
    def start(self):
        for day in range(0, 10):
            self.get_frames_from_dataset(day)
            print("day:", day)
            if type(self.picture_list) != list: continue
            for pic in self.picture_list:
                pic = np.array(pic)
                pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
                yolo_demo(pic)

                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
            print("end of loop")
    # ----------------------not in use------------------------ #

    def find_center_contour(self, contours):    # function gets contour list and return the center contour out of the biggest three in the list

        # take only the mouse contour
        contours.sort(reverse=True, key=cv2.contourArea)
        biggest_three = []

        # find center of image
        image_center = (int(self.only_wound.shape[0]/2),int(self.only_wound.shape[1]/2))
        for i in range(0, 2):
            if (i < len(contours)):
                # find center of each contour
                C = cv2.moments(contours[i])
                if C["m10"] == 0 or C["m00"] == 0 or C["m01"] == 0: continue
                center_Y = int(C["m10"] / C["m00"])
                center_X = int(C["m01"] / C["m00"])
                contour_center = (center_X, center_Y)

                # calculate distance to image_center
                distances_to_center = (distance.euclidean(image_center, contour_center))

                # save to a list of dictionaries
                biggest_three.append({'contour': contours[i], 'distance_to_center': distances_to_center})
        # sort the contours
        sorted_biggest_three = sorted(biggest_three, key=lambda i: i['distance_to_center'])
        return sorted_biggest_three[0]['contour']

    def make_rect_for_grabCut(self,start_p):
        end_p = 1-start_p
        self.rect_for_grab_cut = [(int(self.only_wound.shape[1]*start_p),int(self.only_wound.shape[0]*start_p)),(int(self.only_wound.shape[1]*end_p),int(self.only_wound.shape[0]*end_p))]

    def create_circular_mask(self,h, w, center=None, radius=None):

        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    def grab_cut(self,img,mode="mask"):
        # translate rectangle for grab cut demand
        rect = self.rect_for_grab_cut
        rect_for_grab_cut = [rect[0][0], rect[0][1], rect[1][0], rect[1][1]]

        img_before_segment = img.copy()
        mask = None
        if mode == "rect":
            mode = cv2.GC_INIT_WITH_RECT
            # set mask for the grabCut to update
            mask = np.zeros(img.shape[:2],np.uint8)
            # draw mask on original img
            cv2.rectangle(img_before_segment, rect[0], rect[1], (255, 0, 255), 2)
        else:
            mode = cv2.GC_INIT_WITH_MASK
            # create circular mask
            radius = (self.wound_rect[1][0]-self.wound_rect[0][0])/math.sqrt(2)
            mask = self.create_circular_mask(img.shape[0], img.shape[1], center=None, radius=radius)
            mask = np.where(mask == False, 0, 1).astype('uint8')
            # draw mask on original img
            img_before_segment = cv2.bitwise_and(img_before_segment,img_before_segment,mask = mask)
            # set init values for grab cut mask
            mask[mask > 0] = cv2.GC_PR_FGD
            mask[mask == 0] = cv2.GC_BGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        # use grabCut to mask the background from the segmented objects
        cv2.grabCut(img, mask, rect_for_grab_cut, bgdModel, fgdModel, 100, mode)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]


        return img, img_before_segment, mask2

    def set_square_from_yolo_output(self):
        old_rect_width = self.wound_rect[1][0]-self.wound_rect[0][0]
        old_rect_hight = self.wound_rect[1][1]-self.wound_rect[0][1]
        new_rect_size = max(old_rect_width,old_rect_hight)
        rect_center = (self.wound_rect[0][0]+int(old_rect_width/2),self.wound_rect[0][1]+int(old_rect_hight/2))
        new_rect = [[rect_center[0] - int(new_rect_size/2),rect_center[1]- int(new_rect_size/2)],[rect_center[0] + int(new_rect_size/2),rect_center[1] + int(new_rect_size/2)]]
        # set the square as the new wound rect
        self.wound_rect = new_rect
        return new_rect_size

    def preprocess_frame(self):
        self.cur_frame = np.array(self.cur_frame)
        self.cur_frame = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2RGB)

    def get_rectangle(self):
        self.wound_rect = yolo_demo(self.cur_frame)

    def cut_frame_by_wound(self):
        # cut current frame by wound rect: take the yolo output and make a square with added background, then cut
        bg_percent = 0.3
        new_rect_size = self.set_square_from_yolo_output()
        bg_pixel_to_add = int(bg_percent*new_rect_size)
        self.only_wound = self.cur_frame[-bg_pixel_to_add + self.wound_rect[0][1]:self.wound_rect[1][1] + bg_pixel_to_add, -bg_pixel_to_add + self.wound_rect[0][0]:self.wound_rect[1][0] + bg_pixel_to_add]

    def segment_wound(self):
        # make a rectangle for grab cut algorithm
        self.make_rect_for_grabCut(start_p = 0.22)

        # snake_algorithm(self.only_wound, self.wound_rect)

        # remove hair
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(self.only_wound, cv2.MORPH_CLOSE, kernel, iterations=2)
        # grab cut
        img_grab_cut, img_with_init_mask, wound_mask = self.grab_cut(closing, mode="mask") # mode options are "mask" or "rect"
        cv2.imshow("img_grab_cut", img_grab_cut)
        cv2.imshow("img_grab_cut wound with init mask", img_with_init_mask)
        cv2.waitKey(0)
        contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_contour = self.find_center_contour(contours)
        print("wound area is: ",cv2.contourArea(center_contour))
        segment_wound = self.only_wound.copy()
        cv2.drawContours(segment_wound, [center_contour], -1, (255, 0, 0), 2)
        cv2.imshow("segment wound",segment_wound)
        cv2.waitKey(0)

    def get_wound_segmentation(self, frame=None):
        path = "/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/mouse1.jpg"
        frame = cv2.imread(path)
        self.cur_frame = frame
        self.get_rectangle()
        self.cut_frame_by_wound()
        self.segment_wound()
