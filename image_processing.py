import math

import scipy.signal
from PIL import Image
import cv2
import numpy as np
from scipy.spatial import distance
from scipy.signal import convolve2d
from scipy.ndimage import convolve
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from yolo.yolo_demo import *
from find_wound_playGround import find_squares
from yolo.yolov5_executor import YoloAlgo
import pixellib
from skimage.transform import (hough_line, hough_line_peaks)
import random
from scipy import interpolate
from find_wound_playGround import canny_with_trackbar
import matplotlib
import imutils
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')



class image_process_algo_master:
    def __init__(self, dataset):
        self.dataset = dataset
        self.picture_list = []
        self.cur_frame = None
        self.yolo = YoloAlgo()

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
    def tomer_try_to_Save_the_situation(self,image):
        img = image
        img_shape = img.shape

        x_max = img_shape[0]
        y_max = img_shape[1]

        theta_max = 1.0 * math.pi
        theta_min = 0.0

        r_min = 0.0
        r_max = math.hypot(x_max, y_max)

        r_dim = 200
        theta_dim = 300

        hough_space = np.zeros((r_dim, theta_dim))

        for x in range(x_max):
            for y in range(y_max):
                if img[x, y, 0] == 255: continue
                for itheta in range(theta_dim):
                    theta = 1.0 * itheta * theta_max / theta_dim
                    r = x * math.cos(theta) + y * math.sin(theta)
                    ir = r_dim * (1.0 * r) / r_max
                    hough_space[int(ir), int(itheta)] = hough_space[int(ir), int(itheta)] + 1

        plt.imshow(hough_space, origin='lower')
        plt.xlim(0, theta_dim)
        plt.ylim(0, r_dim)
        plt.show()
    def Harris_corrner_detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > 0.001 * dst.max()] = [0, 0, 255]

        cv2.imshow('dst', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def find_straight_lines(self, image):
        if len(image.shape) != 2: image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        # image = ~image

        # Set a precision of 1 degree. (Divide into 180 data points)
        # You can increase the number of points if needed.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)

        # Perform Hough Transformation to change x, y, to h, theta, dist space.
        hspace, theta, dist = hough_line(image, tested_angles)


        # Now, to find the location of peaks in the hough space we can use hough_line_peaks
        angle_list = []  # Create an empty list to capture all angles

        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(hspace)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(image, cmap='gray')

        origin = np.array((0, image.shape[1]))

        for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist)):
            angle_list.append(angle)  # Not for plotting but later calculation of angles
            a = angle * 180 / np.pi
            if ((a < 95) & (a > 85)):
                y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
                ax[2].plot(origin, (y0, y1), '-r')
        ax[2].set_xlim(origin)
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        plt.tight_layout()
        plt.show()

        ###############################################################
        # Convert angles from radians to degrees (1 rad = 180/pi degrees)
        angles = [a * 180 / np.pi for a in angle_list]

        # Compute difference between the two lines
        angle_difference = np.max(angles) - np.min(angles)
        print(180 - angle_difference)  # Subtracting from 180 to show it as the small angle between two lines
    def prep_img_for_find_lines(self, img1,img2,img3):
        image1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        image2 = ~cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        image3 = ~cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)

        # Invert images to show black background
        # image = ~image  # Invert the image (only if it had bright background that can confuse hough)
        image = cv2.equalizeHist(image1)
        # _, image = cv2.threshold(image, 27, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # image = cv2.Canny(image, 80, 120)
        # kernel = np.ones((3, 3), np.uint8)
        # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)

        # rect_size = 70
        # img_center = (int(image.shape[0]/2),int(image.shape[1]/2))
        # rect = [(img_center[1]-rect_size,img_center[0]-rect_size),(img_center[1]+rect_size,img_center[0]+rect_size)]
        # cv2.rectangle(image, rect[0], rect[1], (255, 255, 255), 2)

        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
        hspace1, theta, dist = hough_line(image, tested_angles)

        # fig, axes = plt.subplots(1, 2, figsize=(30, 15))
        # ax = axes.ravel()
        #
        # ax[0].imshow(image, cmap='gray')
        # ax[0].set_title('Input image 1')
        # # ax[0].set_axis_off()
        #
        # ax[1].imshow(hspace1)
        # ax[1].set_title('Hough transform 1')
        # ax[1].set_xlabel('Angles (degrees)')
        # ax[1].set_ylabel('Distance (pixels)')
        # ax[1].axis('image')
        #
        # plt.show()
        return image
    def find_normaliz_factor_older(self,template,target,ShowMatches=False,ShowTransformImg=False):
        if len(template.shape) != 2: template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        if len(target.shape) != 2: target = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
        # template = self.img_color_equalize(template)
        # target = self.img_color_equalize(target)

        template = cv2.equalizeHist(template)
        target = cv2.equalizeHist(target)

        p1,p2 = self.getPointsSift(template, target, ShowMatches=ShowMatches,R=0.7)
        normalize_factor = self.get_avg_dist_ratio(p1,p2)

        # -----------------try with warp matrix -----------------------
        # H2to1,bestp1 ,bestp2 = self.ransacH(p1, p2, nIter=20, tol=1)
        # warp_shape = self.find_warp_img_shape(templete,H2to1)
        # old_area = target.shape[0]*target.shape[1]
        # new_area = warp_shape[0]*warp_shape[1]
        # normalize_factor = new_area/old_area
        # -------------------------------------------------------------
        if ShowTransformImg:
            # target_warp = cv2.warpPerspective(templete, H2to1, warp_shape)
            print(normalize_factor)
            target_warp = cv2.resize(target,(int(target.shape[1] * normalize_factor), int(target.shape[0] * normalize_factor)))
            cv2.imshow("warp image",target_warp)
            cv2.imshow("target",target)
            cv2.imshow("template",template)
            cv2.waitKey(0)

        self.normaliz_factor = normalize_factor

        return
    def get_normaliz_factor_old(self, ShowLines=False, ShowTransformImg=False, ShowTemplateMatch=False):
        # template match to get a portion of the background
        path = "/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/template.png"
        template = cv2.imread(path)
        rot_mouse = self.rotate_image(self.cur_frame,45)
        target = self.template_match(rot_mouse,template,ShowMatches=ShowTemplateMatch) # find and return square of the background of the current image

        # ---------------exp with backgraound---------------------
        path = "/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/background2.jpeg"
        template = cv2.imread(path)
        template = self.rotate_image(template,90)
        factor = 0.4
        target = cv2.resize(self.cur_frame, (int(self.cur_frame.shape[1] * factor), int(self.cur_frame.shape[0] * factor)))
        norm_fact = self.find_normaliz_factor_older(template,target , ShowMatches=True, ShowTransformImg=True)
        # exit(0)
        # --------------------------------------------------------

        # calc avg distance between squares of the background
        template_dist = self.dist_between_squares(template, ShowLines, imgName="template")
        print("template dist", template_dist)
        target_dist = self.dist_between_squares(target, ShowLines, imgName="target")

        # calc normalize factor between target and template
        normalize_factor = template_dist/target_dist

        if ShowTransformImg:
            print("get_normaliz_factor: normalize_factor=",normalize_factor)
            target_warp = cv2.resize(target,(int(target.shape[1] * normalize_factor), int(target.shape[0] * normalize_factor)))
            cv2.imshow("warp image",target_warp)
            cv2.imshow("target",target)
            cv2.imshow("template",template)
            cv2.waitKey(0)
        self.normaliz_factor = normalize_factor
        return
    def exp_with_energy(self):
        # template match to get a portion of the background
        path = "/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/background2.jpeg"
        template = cv2.imread(path)
        # ----------cutting out the mouse to get background--------------
        mouse, maskBin, mouse_area = self.cut_mouse(self.cur_frame)
        frame_gray = cv2.cvtColor(self.cur_frame,cv2.COLOR_BGR2GRAY)
        tresh = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 15, -2)
        only_bg_tresh = tresh * (~maskBin)
        only_bg_tresh_rot = self.rotate_image(only_bg_tresh, 45)

        # self.find_straight_lines(only_bg_tresh_rot[:temp,:])
        # ---------------------------------------------------------------

        # ------------cutting out a rectangle with max energy---------------
        only_bg_gray = frame_gray * (~maskBin)
        only_bg_gray_rot = self.rotate_image(only_bg_gray, 45)
        rect_size = int(math.sqrt(mouse_area/7))
        bg_rect = self.find_max_energy_rect(only_bg_gray_rot,[rect_size,rect_size])
        bg_rect_gray = bg_rect.copy()
        if len(bg_rect_gray.shape) != 2: bg_rect_gray = cv2.cvtColor(bg_rect_gray, cv2.COLOR_BGR2GRAY)
        # ---------------------------------------------------------------
        # template match
        self.scale_template_match(bg_rect_gray,tresh=0.6)

        #----------------------------------- Canny -------------------------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bg_rect_gray_blur = cv2.blur(bg_rect_gray,(9,9))
        bg_rect_H = cv2.cvtColor(bg_rect,cv2.COLOR_BGR2HSV)[:,:,0]
        cv2.imshow("bg_rect_H", bg_rect_H)
        bg_rect_S = cv2.cvtColor(bg_rect,cv2.COLOR_BGR2HSV)[:,:,1]
        cv2.imshow("bg_rect_S", bg_rect_S)
        bg_rect_V = cv2.cvtColor(bg_rect,cv2.COLOR_BGR2HSV)[:,:,2]
        cv2.imshow("bg_rect_V", bg_rect_V)
        bg_rect_H_blur = cv2.blur(bg_rect_V, (3, 3))
        bg_rect_gray_eq = cv2.equalizeHist(bg_rect_gray)
        bg_rect_tresh_eq = cv2.adaptiveThreshold(bg_rect_gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 15, -2)
        # cv2.imshow("bg equaliz",bg_rect_gray_eq)
        # cv2.imshow("bg_rect_tresh_eq",bg_rect_tresh_eq)
        # cv2.imshow("bg_rect_gray_eq",bg_rect_gray_eq)
        # canny = canny_with_trackbar(bg_rect_gray_eq)

        # canny_rect = cv2.Canny(bg_rect_H_blur, 0, 3, 3)
        # cv2.imshow("bg_rect_H", bg_rect_H_blur)

        # lines =cv2.HoughLinesP(canny_rect,1,np.pi/180,10)
        # print(lines[0][0])
        # cv2.line(bg_rect,(lines[0][0][0],lines[0][0][1]),(lines[0][0][2],lines[0][0][3]),(0,255,0),5)
        # cv2.imshow("lines",bg_rect)
        #--------------------------------------------------------------------------------------------------
        return
        # self.dist_between_squares(bg_rect, ShowLines=True, imgName="target")
        self.find_straight_lines(bg_rect_tresh)
    def find_horz_and_vert_lines(self,img):

        # Transform source image to gray if it is not already
        if len(img.shape) != 2: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
        img_not = cv2.bitwise_not(img)
        bw = cv2.adaptiveThreshold(img_not, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        # Create the images that will use to extract the horizontal and vertical lines
        horizontal = np.copy(bw)
        vertical = np.copy(bw)

        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = int(cols/50)
        print("horizontal_size:",horizontal_size)
        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=2)

        self.find_straight_lines(horizontal)


        # Show extracted horizontal lines
        # cv2.imshow("horizontal", horizontal)
        # cv2.waitKey(0)

        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)
        # Show extracted horizontal lines
        # cv2.imshow("horizontal", horizontal)
        # cv2.waitKey(0)

        # Specify size on vertical axis
        rows = vertical.shape[0]
        verticalsize = int(rows/50)
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        self.find_straight_lines(vertical)

        # Show extracted vertical lines
        # cv2.imshow("vertical", vertical)
        # cv2.waitKey(0)
        return
    def segment_using_hsv_thresh(self,img):
        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H = image_hsv[:, :, 0]
        S = image_hsv[:, :, 1]
        V = image_hsv[:, :, 2]

        mask = np.zeros(image_hsv.shape[:2], np.uint8)
        maskIndex = np.where( (45 > H) & (25 > S))
        mask[maskIndex] = 255
        maskBin = (np.array(mask) > 0)

        mask_dilate = ~cv2.dilate(~mask, (5,5), iterations=20)
        maskBin_dilate = (np.array(mask_dilate) > 0)

        masked_img = img * maskBin[:, :, np.newaxis]
        cv2.imshow("img",img)
        cv2.imshow("mask",mask)
        cv2.imshow("mask_dilate",mask_dilate)
        cv2.waitKey(0)
        return masked_img, maskBin, maskBin_dilate
    def get_histogram(self, image):
        """
         * Python program to create a color histogram.
         *
         * Usage: python ColorHistogram.py <filename>
        """
        import sys
        import skimage.io
        from matplotlib import pyplot as plt

        # read original image, in full color, based on command
        # line argument
        image = np.array(image)
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv image", hsv_frame)
        cv2.imshow("rgb image", rgb_frame)
        # tuple to select colors of each channel line
        colors = ("red", "green", "blue")
        channel_ids = (0, 1, 2)

        # create the histogram plot, with three lines, one for
        # each color
        plt.xlim([0, 256])
        for channel_id, c in zip(channel_ids, colors):
            histogram, bin_edges = np.histogram(
                hsv_frame[:, :, channel_id], bins=256, range=(0, 256)
            )
            plt.plot(bin_edges[0:-1], histogram, color=c)

        plt.xlabel("Color value")
        plt.ylabel("Pixels")

        plt.show()
        key = cv2.waitKey(0)
    def img_color_equalize(self,img):
        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

        # convert back to RGB color-space from YCrCb
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

        # cv2.imshow('Color input image', img)
        # cv2.imshow('Histogram equalized', img_output)
        # cv2.waitKey(0)
        return equalized_img

    # ---------------warp image (not in use)------------------ #
    def corrMat(self,p1, p2):
        '''
        p1 and p2 - each of them is Nx2 numpy array ordered by matching
        :return: 2 rows of the A matrix for computing H2to1
        '''
        r0 = np.concatenate((-p2.T, -np.ones((1, 1)), np.zeros((1, 3))), axis=1)
        r1 = np.concatenate((np.zeros((1, 3)), -p2.T, -np.ones((1, 1))), axis=1)
        p_part = np.concatenate((r0, r1))
        return np.concatenate((p_part, (p2 @ p1.T).T, p1), axis=1)
    def getA(self,p1, p2):
        '''
        p1 and p2 - each of them is Nx2 numpy array ordered by matching
        :return: A matrix for computing H2to1
        '''
        A = np.zeros((0, 9))
        for pp1, pp2 in zip(p1, p2):
            pp1, pp2 = pp1.reshape(2, 1), pp2.reshape(2, 1)
            p_mat = self.corrMat(pp1, pp2)
            A = np.concatenate((A, p_mat)).astype('int32')
        return A
    def computeH(self,p1, p2):
        '''
        p1 and p2 - each of them is Nx2 numpy array ordered by matching
        :return: H2to1
        '''
        A = self.getA(p1, p2)
        u, s, v = np.linalg.svd(A, full_matrices=True)
        H2to1 = v[-1, :].reshape(3, 3)
        return H2to1
    def interpolation(self,image):
        interp_func = []
        ch_num = 3
        if len(image.shape) < 3: ch_num = 1
        for ch in range(0, ch_num):
            y_indices = np.arange(image.shape[0])
            x_indices = np.arange(image.shape[1])
            interp_func.append(interpolate.interp2d(x_indices, y_indices, image[:, :, ch], kind='linear'))
        return interp_func
    def checkBounds(self, x, y, img):
        height, width = img.shape[0], img.shape[1]
        return (y < 0 or y > height - 1 or x < 0 or x > width - 1)
    def warpH(self,im1, H, out_size):
        # get offset from img1
        im1Corners = [np.array([0, 0, 1]).T, np.array([0, im1.shape[0], 1]).T, np.array([im1.shape[1], 0, 1]).T,
                      np.array([im1.shape[1], im1.shape[0], 1]).T]
        im2Corners = []
        for p in im1Corners:
            im2Corners.append((H @ p) / (H @ p)[2])
        pointsX = [p[0] for p in im2Corners]
        pointsY = [p[1] for p in im2Corners]
        Xmin, Xmax = min(pointsX), max(pointsX)
        Ymin, Ymax = min(pointsY), max(pointsY)
        Xoffset, Yoffset = 0, 0
        if Xmin < 0:
            Xoffset = Xmin
        if Xmax > out_size[1]:
            Xoffset = -(Xmax - out_size[1])
        if Ymin < 0:
            Yoffset = Ymin
        if Ymax > out_size[0]:
            Yoffset = -(Ymax - out_size[0])
        H1to2 = np.linalg.inv(H)
        warp_im1 = np.zeros((out_size[0], out_size[1], 3))
        interp_func = self.interpolation(im1)

        for y in range(out_size[0]):
            for x in range(out_size[1]):
                p_im1 = np.array([x + Xoffset, y + Yoffset, 1]).T
                p_im1 = H1to2 @ p_im1
                p_im1 = (p_im1 / (p_im1[2] + 1e-10))
                if self.checkBounds(p_im1[0], p_im1[1], im1):
                    continue  # pixel value stay zero
                for ch in range(0, len(interp_func)): warp_im1[y][x][ch] = interp_func[ch](p_im1[0], p_im1[1])[0]
        return warp_im1.astype('uint8')
    def get_corners(self,image):
        corners = [np.array([0, 0, 1]).T, np.array([0, image.shape[0], 1]).T, np.array([image.shape[1], 0, 1]).T,
                   np.array([image.shape[1], image.shape[0], 1]).T]
        return corners
    def find_warp_img_shape(self,image,HMatrix):
        img_corners = self.get_corners(image)
        img_corners_after_H = [(HMatrix @ p)/(HMatrix @ p)[2] for p in img_corners]
        x_list = [p[0] for p in img_corners_after_H]
        y_list = [p[1] for p in img_corners_after_H]
        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)
        new_width = int(x_max-x_min)+1
        new_length = int(y_max-y_min)+1
        return (new_width,new_length)
    def computeA(self, p2, p1):
        A = np.zeros((0, 9))
        for point1, point2 in zip(p1, p2):
            point1, point2 = point1.reshape(1, 2), point2.reshape(1, 2)
            row0 = np.concatenate((point1, np.ones((1, 1)), np.zeros((1, 3))), axis=1)
            row0 = np.concatenate((row0, -(point1 * point2[0][0]), [[-point2[0][0]]]), axis=1)
            row1 = np.concatenate((np.zeros((1, 3)), point1, np.ones((1, 1))), axis=1)
            row1 = np.concatenate((row1, -(point1 * point2[0][1]), [[-point2[0][1]]]), axis=1)
            A_part = np.concatenate((row0, row1))
            A = np.concatenate((A, A_part)).astype('int32')
        return A
    def is_inliner(self, pp1, pp2, H2to1, tol):
        pp2to1 = (H2to1 @ np.array((pp2[0], pp2[1], 1))) / (H2to1 @ np.array((pp2[0], pp2[1], 1)))[2]
        dist = (pp2to1[0] - pp1[0]) + (pp2to1[1] - pp1[1])
        if dist < tol:
            return True
        return False
    def countMetches(self, p1, p2, tempH, tol):
        count = 0
        tempP1 = []
        tempP2 = []
        for point1, point2 in zip(p1, p2):
            if self.is_inliner(point1, point2, tempH, tol):
                tempP1.append(point1), tempP2.append(point2)
                count += 1
        return count, tempP1, tempP2
    def ransacH(self, p1, p2, nIter=20, tol=1):
        if p1.shape[0] < 4:  # check num of matches is sufficient
            raise Exception("less then 4 matches")
        maxPoints = 0
        for i in range(nIter):
            four_random_indices = [random.randrange(p1.shape[0]) for _ in range(4)]
            randomP1 = np.array([p1[index] for index in four_random_indices])
            randomP2 = np.array([p2[index] for index in four_random_indices])
            tempH , _ =  cv2.findHomography(randomP1, randomP2)
            count, tempP1, tempP2 = self.countMetches(p1, p2, tempH, tol)
            if count > maxPoints:
                bestP1, bestP2 = tempP1, tempP2
                maxPoints = count
        bestP1 = np.asarray(bestP1)
        bestP2 = np.asarray(bestP2)
        print("ransacH",len(bestP2), len(bestP1))
        bestH , _ =  cv2.findHomography(bestP1, bestP2)
        return bestH , bestP1, bestP2
    def filterOutBadMatches(self,im1, im2, kp1, kp2, matches, R, ShowMatches=False):
        goodMatches = []
        for p1, p2 in matches:
            if p1.distance < R * p2.distance:
                goodMatches.append([p1])
        p1, p2 = [], []
        for match in goodMatches:
            p1.append(kp1[match[0].queryIdx].pt)
            p2.append(kp2[match[0].trainIdx].pt)

        if ShowMatches:
            img3 = cv2.drawMatchesKnn(im1, kp1, im2, kp2, goodMatches, None, flags=2)
            cv2.startWindowThread()
            cv2.imshow("matches", img3)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            key = cv2.waitKey(1)

        p1, p2 = np.array(p1), np.array(p2)
        return p1, p2
    def getPointsSift(self,im1, im2, ShowMatches=False, R=0.4):
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(im1, None)
        kp2, des2 = sift.detectAndCompute(im2, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        print("getPointsSift: num of matches is: ",len(matches))
        p1, p2 = self.filterOutBadMatches(im1, im2, kp1, kp2, matches, R, ShowMatches)
        print("getPointsSift: num of filterde matches is: ",len(p1))
        return p1, p2

    # ----------------- get normalization factor --------------------------
    def get_contours(self, pic):
        blur = cv2.GaussianBlur(pic, (5, 5), 0)
        _, thresh_pic = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        close = cv2.morphologyEx(thresh_pic, cv2.MORPH_CLOSE, kernel, iterations=5)
        close[close.shape[0]-20:close.shape[0]-1] = 0
        contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def cut_mouse(self, pic):
        # find mouse contour
        gray_pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        orig_contours = self.get_contours(gray_pic)
        center_contour = []
        center_contour.append(self.find_center_contour(orig_contours, pic=pic))

        # cut out the mouse
        pic_copy = pic.copy()

        mask = np.zeros(pic_copy.shape, np.uint8)
        cv2.drawContours(mask, center_contour,-1, (0,255,0), thickness=-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # close the contour of the mouse
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
        mouse_contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mouse_area = cv2.contourArea(mouse_contours[0])

        mouse = pic.copy()
        cv2.drawContours(mouse, mouse_contours, -1, (0, 0, 255), thickness=-1)

        # create mask
        mask = np.zeros(pic.shape, np.uint8)
        cv2.drawContours(mask, mouse_contours,-1, (0,0,255), thickness=-1)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        mouse_gray_dilate = cv2.dilate(mask,kernel, iterations=10)
        mask_out = (np.array(mouse_gray_dilate) > 0)

        return mouse , mask_out, mouse_area

    def find_max_energy_rect(self,frame,rect_size,ShowBgRect=False):
        color_frame = self.rotate_image(self.cur_frame,45)
        frame_gray = frame.copy()
        if len(frame_gray.shape) != 2: frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)
        kernal = (rect_size[0],rect_size[1])
        conv_output = cv2.blur(frame_gray,kernal)
        conv_output = conv_output[int(rect_size[0]/2):frame_gray.shape[0]-int(rect_size[0]/2),int(rect_size[1]/2):frame_gray.shape[1]-int(rect_size[1]/2)]
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(conv_output)
        rect_corner = [maxLoc[0],maxLoc[1]]
        max_rect = frame[rect_corner[1]:rect_corner[1]+rect_size[0],rect_corner[0]:rect_corner[0]+rect_size[0]]
        max_rect_color = color_frame[rect_corner[1]:rect_corner[1]+rect_size[0],rect_corner[0]:rect_corner[0]+rect_size[0]]

        if ShowBgRect:
            frame_with_rect = frame_gray.copy()
            cv2.rectangle(frame_with_rect, rect_corner, (rect_corner[0] + rect_size[0], rect_corner[1] + rect_size[1]), (0, 0, 255), 2)
            cv2.imshow("frame with rect",frame_with_rect)
            cv2.imshow("max energy rect",max_rect_color)

        return max_rect_color

    def scale_template_match(self,image, tresh=0.7, ShowMatch=False):
        def prep_templates():
            template0 = cv2.imread("/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/template_smaller0.png")
            template1 = cv2.imread("/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/template_smaller1.png")
            template3 = cv2.imread("/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/template_smaller3.png")
            templates = [template0, template1, template3]
            for i in range(0,len(templates)): templates[i] = cv2.equalizeHist(cv2.cvtColor(templates[i], cv2.COLOR_BGR2GRAY))
            return templates
        def show_match(match):
            val, loc, w, h = match
            img2 = image.copy()
            bottom_right = (loc[0] + w, loc[1] + h)
            cv2.rectangle(img2, loc, bottom_right, 0, 5)
            cv2.imshow("match", img2)
            print("sale_template_match: confidence=",val,"w=",w)

        templates = prep_templates()
        print("start template match")
        image_gray = image.copy()
        if len(image_gray.shape) != 2: image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.equalizeHist(image_gray)

        best_match = [0, None, None]
        for template in templates:
            matches = []
            for scale in np.linspace(0.3, 2, 30)[::-1]:
                w = int(template.shape[1] * scale)
                h = int(template.shape[0] * scale)
                resized = cv2.resize(template, (w,h))
                if resized.shape[0] <= image.shape[0] and resized.shape[1] <= image.shape[1]:
                    res = cv2.matchTemplate(image_gray, resized, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if(max_val >= tresh):
                        match = [max_val,max_loc,w,h]
                        matches.append(match)

            for match in matches:
                if match[0] > best_match[0]: best_match = match
        print("finished template match")
        if best_match[0] > 0:
            if ShowMatch: show_match(best_match)
            return best_match[2]*best_match[3]  # square area
        else:
            print("no match found")
            return None

    def dist_between_squares(self, img, ShowLines=False, imgName="template"):  # FIXME not good enough
        lines_position = self.find_horz_lines(img)
        lines_position.sort()
        dist1 = lines_position[2] - lines_position[1]
        dist2 = lines_position[1] - lines_position[0]
        if max(dist2,dist1)/min(dist2,dist1) > 1.2: avg_dist = max(dist1, dist2)
        else: avg_dist = (dist1 + dist2)/2
        if ShowLines:
            if len(img.shape) != 2: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for line in lines_position: img[line] = 0
            print("dist_between_squares: dist1=",dist1,"dist2=", dist2)
            print("dist_between_squares: avg dist=",avg_dist)
            print("rect height",img.shape[0])

            cv2.imshow(imgName,img)
            cv2.waitKey(0)
        return avg_dist

    def find_horz_lines(self, img):

        # Transform source image to gray if it is not already
        if len(img.shape) != 2: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
        img_not = cv2.bitwise_not(img)
        img = cv2.adaptiveThreshold(img_not, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        max_line_width = int(img.shape[0]*0.08)
        sum_list = []
        lines_idx = []
        for line_num, line in enumerate(img):
            Sum = sum(line)
            sum_list.append(Sum)
        for i in range(0,3):
            max_value = max(sum_list)
            max_index = sum_list.index(max_value)
            lines_idx.append(max_index)
            for i in range(0,max_line_width):
                index_to_erase = max_index + int(max_line_width/2) - i
                if index_to_erase >= 0 and index_to_erase < len(sum_list):
                    sum_list[index_to_erase] = 0

        return lines_idx

    def rotate_image(self, image,degrees):
        rotated = imutils.rotate_bound(image, degrees)
        return rotated

    def get_avg_dist_ratio(self, bestp1, bestp2):
        def calc_dist(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        dist_list = []
        for i in range(0,bestp1.shape[0]):
            p1_1 = bestp1[i]
            p2_1 = bestp2[i]
            for j in range(i+1, bestp1.shape[0]):
                p1_2 = bestp1[j]
                p2_2 = bestp2[j]
                dist_in_1 = calc_dist(p1_1,p1_2)
                dist_in_2 = calc_dist(p2_1,p2_2)
                if dist_in_1 == 0 or dist_in_2 == 0: continue
                dist_list.append(dist_in_1/dist_in_2)   # save the ratio between the distances
        avg_ratio = sum(dist_list) / len(dist_list)
        return avg_ratio

    def template_match(self, img, template, ShowMatches=False):
        def show_match(loc,title):
            bottom_right = (loc[0] + w, loc[1] + h)
            cv2.rectangle(img2, loc, bottom_right, 255, 5)
            cv2.imshow(title, img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        def find_best(locations):
            best = locations[0]
            best_count = 1
            for i in range(0,len(locations)):
                count = 1
                for j in range(i+1, len(locations)):
                    if locations[i][0] == locations[j][0] and locations[i][1] == locations[j][1]: count = count + 1
                if count > best_count:
                    best = locations[i]
                    best_count = count
            return best
        print("start template match")


        h, w = template.shape[0], template.shape[1]
        methods = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCOEFF_NORMED, cv2.TM_CCOEFF, cv2.TM_CCORR_NORMED, cv2.TM_CCORR]
        locations = []
        for method in methods:
            img2 = img.copy()
            result = cv2.matchTemplate(img2, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                location = min_loc
            else:
                location = max_loc
            locations.append(location)
            if ShowMatches: show_match(location,str(method))
        best_match = find_best(locations)
        if ShowMatches:
            img2 = img.copy()
            show_match(best_match,"best")
        print("finished template match")
        return img[best_match[1]:best_match[1] + w, best_match[0]:best_match[0]+ h]

    def get_normaliz_factor(self,ShowBgRect=False, ShowTemplateMatch=False, ShowTransformImg=False):
        # ----------cutting out the mouse to get background--------------
        mouse, maskBin, mouse_area = self.cut_mouse(self.cur_frame)
        frame_gray = cv2.cvtColor(self.cur_frame,cv2.COLOR_BGR2GRAY)

        # ------------cutting out a rectangle with max energy---------------
        only_bg_gray = frame_gray * (~maskBin)
        only_bg_gray_rot = self.rotate_image(only_bg_gray, 45)
        rect_size = int(math.sqrt(mouse_area/7))
        bg_rect = self.find_max_energy_rect(only_bg_gray_rot,[rect_size,rect_size], ShowBgRect)
        bg_rect_gray = bg_rect.copy()
        if len(bg_rect_gray.shape) != 2: bg_rect_gray = cv2.cvtColor(bg_rect_gray, cv2.COLOR_BGR2GRAY)
        # ---------------------------------------------------------------
        # template match
        template = cv2.imread("/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/template_smaller0.png")
        template_area = template.shape[0]*template.shape[1]
        template_dist = math.sqrt(template_area)
        square_area = self.scale_template_match(bg_rect_gray,tresh=0.6, ShowMatch=ShowTemplateMatch)
        if square_area is not None:
            normalize_factor = template_dist/math.sqrt(square_area)
            self.normaliz_factor = normalize_factor
        else:
            print("no normaliz factor found")
            self.normaliz_factor = None
            return False
        if ShowTransformImg:
            print("get_normaliz_factor: normalize_factor=",normalize_factor)
            target_warp = cv2.resize(bg_rect,(int(bg_rect.shape[1] * normalize_factor), int(bg_rect.shape[0] * normalize_factor)))
            cv2.imshow("warp image",target_warp)
            cv2.imshow("target",bg_rect)
            cv2.imshow("template",template)
        if ShowBgRect or ShowTemplateMatch or ShowTransformImg:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return True
    # ---------------------------------------------------------------------

    def set_wound_area_in_dataset(self):  # FIXME need to test
        normalized_area = self.cur_wound_area*self.normaliz_factor
        self.dataset.update_mouse_in_dataset(self.cur_mouse_name, self.cur_day, normalized_area, type="area")
        return

    def set_bound_circle_r_in_dataset(self): # FIXME need to test
        normalized_radius = self.cur_bounding_radius*self.normaliz_factor
        self.dataset.update_mouse_in_dataset(self.cur_mouse_name, self.cur_day, normalized_radius, type="radius")
        return

    def get_last_wound_area(self):
        if self.normaliz_factor is None: return None
        last_day = self.dataset.get_last_day(self.cur_mouse_name,self.cur_day)
        if last_day == None: return None
        pic = self.dataset.get_pic_with_tag(self.cur_mouse_name, last_day)
        if pic is None:
            return None
        else:
            return pic.algo_size_in_pixels/self.normaliz_factor

    def get_last_bounding_radius(self):
        if self.normaliz_factor is None: return None
        last_day = self.dataset.get_last_day(self.cur_mouse_name,self.cur_day)
        if last_day == None: return None
        pic = self.dataset.get_pic_with_tag(self.cur_mouse_name, last_day)
        if pic is None:
            return None
        else:
            return pic.min_bounding_radius_in_pixels/self.normaliz_factor

    def find_min_bounding_circle(self, contour,img=None,display=False):
        (x,y), r = cv2.minEnclosingCircle(contour)
        center_x = int(x)
        center_y = int(y)
        r = int(r)
        if display:
            cv2.circle(img, (center_x, center_y), r, (255, 255, 0), 2)
            cv2.imshow("min bounding circle", img)
            cv2.waitKey(0)
        return r

    def find_center_contour(self, contours, pic=None):    # function gets contour list and return the center contour out of the biggest three in the list

        # take only the mouse contour
        contours.sort(reverse=True, key=cv2.contourArea)
        biggest_three = []
        if pic is None: img = self.only_wound
        else: img = pic
        # find center of image
        image_center = (int(img.shape[0]/2),int(img.shape[1]/2))
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
            radius = None
            if self.last_bounding_radius:
                radius = int(self.last_bounding_radius*1.1)
            else:
                radius = int(self.wound_rect_size/2)
                self.last_bounding_radius = radius
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

    def set_square_from_yolo_output(self,bounding_r=None):  # bounding_r: the radius that was used in the last iteration
        old_rect_width = self.wound_rect[1][0]-self.wound_rect[0][0]
        old_rect_hight = self.wound_rect[1][1]-self.wound_rect[0][1]
        rect_center = (self.wound_rect[0][0]+int(old_rect_width/2),self.wound_rect[0][1]+int(old_rect_hight/2))

        if bounding_r:
            new_rect_size = bounding_r*2
        else:
            new_rect_size = max(old_rect_width,old_rect_hight)
        new_rect = [[rect_center[0] - int(new_rect_size/2),rect_center[1]- int(new_rect_size/2)],[rect_center[0] + int(new_rect_size/2),rect_center[1] + int(new_rect_size/2)]]
        # set the min square that contain the wound as the new wound rect
        self.wound_rect = new_rect
        self.wound_rect_size = new_rect_size
        self.last_bounding_radius = bounding_r
        print("set_square_from_yolo_output: use r= ",bounding_r)

    def get_rectangle(self):
        yolov5_detection = self.yolo.run(self.cur_frame)
        self.wound_rect = None if yolov5_detection is None else [[yolov5_detection['x0'], yolov5_detection['y0']], [yolov5_detection['x1'], yolov5_detection['y1']]]

    def cut_frame_by_wound(self,bounding_r=None):  # bounding_r: the radius that was used in the last iteration
        # cut current frame by wound rect: take the yolo output and make a square with added background, then cut
        bg_percent = 0.3
        self.set_square_from_yolo_output(bounding_r)
        bg_pixel_to_add = int(bg_percent*self.wound_rect_size)
        self.only_wound = self.cur_frame[-bg_pixel_to_add + self.wound_rect[0][1]:self.wound_rect[1][1] + bg_pixel_to_add, -bg_pixel_to_add + self.wound_rect[0][0]:self.wound_rect[1][0] + bg_pixel_to_add]

    def segment_wound(self):
        # make a rectangle for grab cut algorithm
        self.make_rect_for_grabCut(start_p = 0)

        # remove hair
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(self.only_wound, cv2.MORPH_CLOSE, kernel, iterations=2)
        closing = self.only_wound #FIXME delete!!!!
        # grab cut
        img_grab_cut, img_with_init_mask, wound_mask = self.grab_cut(closing, mode="mask") # mode options are "mask" or "rect"
        cv2.imshow("img_grab_cut", img_grab_cut)
        cv2.imshow("img_grab_cut wound with init mask", img_with_init_mask)
        cv2.waitKey(0)
        contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_contour = self.find_center_contour(contours)
        wound_area = cv2.contourArea(center_contour)
        print("segment_wound: wound area is=",wound_area)
        only_wound_to_show = self.only_wound.copy()
        cv2.drawContours(only_wound_to_show, [center_contour], -1, (255, 0, 0), 2)
        cv2.imshow("segmented wound",only_wound_to_show)
        cv2.waitKey(0)
        # prepare bounding radius for net image evaluation
        circle_r = self.find_min_bounding_circle(center_contour, only_wound_to_show,display=True)
        cv2.destroyAllWindows()
        return circle_r, wound_area

    def get_wound_segmentation(self, mouse_full_name=None, day=None):
        path = "/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/bounding circle exp/day0.jpg"

        pics = self.dataset.get_pic_with_tag(mouse_full_name,day)
        if pics is None: return
        pics = pics.pictures
        for pic in pics:
            frame = cv2.imread(pic)
            # preper data for frame segmentation
            self.cur_frame = frame
            self.cur_mouse_name = mouse_full_name
            self.cur_day = day
            if not self.get_normaliz_factor(ShowBgRect=True, ShowTransformImg=True, ShowTemplateMatch=True): return


            last_bound_circle_r = self.get_last_bounding_radius()
            last_wound_area = self.get_last_wound_area()
            wound_area = None

            # start segmentation algorithm
            self.get_rectangle()
            if self.wound_rect is None:
                return
            while wound_area is None or wound_area > last_wound_area:
                self.cut_frame_by_wound(bounding_r=last_bound_circle_r) # bounding_r: the radius that was used in the last iteration
                bound_circle_r, wound_area = self.segment_wound()
                last_bound_circle_r = int(self.last_bounding_radius * 0.9)  # decries radius for next iteration
                if last_wound_area is None: break
            self.cur_bounding_radius = bound_circle_r
            self.cur_wound_area = wound_area

            self.set_wound_area_in_dataset()
            self.set_bound_circle_r_in_dataset()

