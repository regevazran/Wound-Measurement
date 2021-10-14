import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from yolo.yolo_demo import *
from find_wound import grab_cut
import pixellib


class image_process_algo_master:
    def __init__(self, dataset):
        self.dataset = dataset
        self.picture_list = []
        self.cur_frame = None

    def get_frames_from_dataset(self, day=0):
        self.picture_list = self.dataset.get_pic_with_tag(mouse_name=self.dataset.data["Mouse"][0], day=day).pictures

    def pic2HSV(self, pic):
        pic = np.array(pic) # FIXME check if needed
        pic = cv2.resize(pic, (int(pic.shape[1]*0.4), int(pic.shape[0]*0.4))) # FIXME check if needed
        hsv_frame = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
        pic_rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        # min_H = hsv_frame[..., 0].min()
        # min_S = hsv_frame[..., 1].min()
        # min_V = hsv_frame[..., 2].min()
        #
        # max_H = hsv_frame[..., 0].max()
        # max_S = hsv_frame[..., 1].max()
        # max_V = hsv_frame[..., 2].max()
        # print("min max values are: (",min_H, max_H,",",min_S, max_S,",",min_V, max_V,")")
        # define range of red color in HSV
        # lower_red = np.array([115, 42, 55])
        # upper_red = np.array([155, 170, 188])
        lower_red = np.array([107, 14, 36])
        upper_red = np.array([150, 153, 171])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv_frame, lower_red, upper_red)
        cv2.imshow("mask",mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=5)
        # open = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel,iterations=5)

        # Bitwise-AND mask and original image
        thresh_pic = cv2.bitwise_and(pic_rgb, pic_rgb, mask=close)
        cv2.imshow("pic_rgb",pic_rgb)
        cv2.imshow("thresh_pic",thresh_pic)

        s = np.linspace(0, 2 * np.pi, 400)
        r = 100 + 100 * np.sin(s)
        c = 220 + 100 * np.cos(s)
        init = np.array([r, c]).T
        print(init.shape)
        print(init)
        return hsv_frame

    def find_center_contour(self, contours, pic):

        # take only the mouse contour
        contours.sort(reverse= True,key = cv2.contourArea)
        biggest_three = []

        # find center of image
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
        blur = cv2.GaussianBlur(pic, (5, 5), 0)
        _, thresh_pic = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow("thresh", thresh_pic)
        contours, _ = cv2.findContours(thresh_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    def cut_mouse(self,pic):
        # find mouse contour
        gray_pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        orig_contours = self.get_contours(gray_pic)
        center_contour = []
        center_contour.append(self.find_center_contour(contours=orig_contours, pic= pic))

        # cut out the mouse
        pic_copy = pic.copy()
        cv2.drawContours(pic_copy, center_contour, -1, (0, 255, 0), thickness=-1)
        mask = (pic_copy == (0,255,0))
        pic_copy = pic*mask
        gray_pic = cv2.cvtColor(pic_copy, cv2.COLOR_BGR2GRAY)
        gray_pic[np.where((gray_pic > 0))] = 255

        #close the contour of the mouse
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        close = cv2.morphologyEx(gray_pic, cv2.MORPH_CLOSE, kernel,iterations=5)
        mouse_contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pic_copy = pic.copy()
        cv2.drawContours(pic_copy, mouse_contours, -1, (0, 255, 0), thickness=-1)
        mask = (pic_copy == (0,255,0))
        pic_copy = pic*mask

        # # cut a rectangle around the mouse
        # x, y, w, h = cv2.boundingRect(contour[0])
        # pic_copy = pic.copy()
        # pic_copy[0:y,:] = (255,255,255)
        # pic_copy[:,0:x] = (255,255,255)
        # pic_copy[y+h:pic_copy.shape[0],:] = (255,255,255)
        # pic_copy[:,x + w: pic_copy.shape[1]] = (255,255,255)
        return pic_copy

    def kmeans_clostoring(self,pic):

        Z = pic.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((pic.shape))
        cv2.imshow('res2', res2)
        cv2.waitKey(0)

    def spectral_clustering(self, img):
        mask = img.astype(bool)
        img = img.astype(float)

        img += 1 + 0.2 * np.random.randn(*img.shape)
        print("start image to graph")

        graph = image.img_to_graph(img, mask=mask)
        print("finished image to graph")

        graph.data = np.exp(-graph.data / graph.data.std())
        print("start spectral_cluster")

        labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
        print("finished spectral_cluster")

        label_im = np.full(mask.shape, -1.)
        label_im[mask] = labels

        cv2.imshow("img", img)
        cv2.imshow("label_im", label_im)
        cv2.waitKey(0)

        return

    def deep_segmentation(self,pic):
        path = '/Users/regevazran/Desktop/technion/semester i/project c/temp pic/'
        result = cv2.imwrite(path+'temp_pic.jpg', pic)
        from pixellib.semantic import semantic_segmentation

        segment_image = semantic_segmentation()
        segment_image.load_pascalvoc_model(path+"deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
        segment_image.segmentAsPascalvoc(path+'temp_pic.jpg', output_image_name=path+"image_new.jpg")
        return

    def get_wound_contour(self, pic):
        blur = cv2.GaussianBlur(pic, (5, 5), 0)
        edges = cv2.Canny(blur, 20, 70 )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("edges", edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def preprocess_pic(self, pic):
        pic = np.array(pic)
        cv2.imshow("original pic", cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))  # FIXME delete
        pic = cv2.resize(pic, (int(pic.shape[1]*0.4), int(pic.shape[0]*0.4)))

        cropped = self.cut_mouse(pic)

        # self.kmeans_clostoring(cropped)
        # self.spectral_clustering(cropped)
        # self.deep_segmentation(cropped)

        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # wound_contours = self.get_wound_contour(gray_cropped)
        # for i in range(len(wound_contours)):
        #     hull = cv2.convexHull(wound_contours[i])
        #     cv2.drawContours(cropped, [hull], -1, (255, 0, 0), -1)
        # cv2.drawContours(cropped, wound_contours, -1, (0, 255, 0), thickness=cv2.FILLED)
        # cv2.imshow("cropped", cropped)
        # cv2.waitKey(0)

        ## find wound contour ##
        # contours = self.get_contours(gray_cropped)
        # cropped_center_contour = []
        # cropped_center_contour.append(self.find_center_contour(contours= contours, pic= pic))
        # cv2.drawContours(cropped, cropped_center_contour, -1, (0, 255, 0), thickness=3)
        # cv2.imshow("wound contour", cropped)
        # cv2.waitKey(0)
        return cropped

    def get_wound_size(self, preprocess_pic):
        pass

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
        cv2.imshow("hsv image",hsv_frame)
        cv2.imshow("rgb image",rgb_frame)
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

    def histogram_equalization(self, image):

        # convert from RGB color-space to YCrCb
        image = np.array(image)
        image = cv2.resize(image, (int(image.shape[1]*0.4), int(image.shape[0]*0.4)))
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

        # convert back to RGB color-space from YCrCb
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

        # cv2.imshow('img no HE', image)
        # cv2.imshow('equalized_img', equalized_img)
        # key = cv2.waitKey(0)
        return equalized_img

    # def start(self):
    #     for day in range(0, 10):
    #         self.get_frames_from_dataset(day)
    #         print("day:", day)  # FIXME delete
    #         if type(self.picture_list) != list: continue
    #         for pic in self.picture_list:
    #             pic = np.array(pic)
    #             pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    #             yolo_demo(pic)
    #
    #             key = cv2.waitKey(0)
    #             if key == ord('q'):
    #                 break
    #         print("end of loop")

    def segment_wound(self):
        grab_cut(self.cur_frame, self.wound_rect)

    def cut_frame_by_wound(self):
        # cut cur frame by wound rect
        self.only_wound = None

    def get_rectangle(self):
        self.wound_rect = yolo_demo(self.cur_frame)


    def preprocess_frame(self):
        self.cur_frame = np.array(self.cur_frame)
        self.cur_frame = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2RGB)

    def get_wound_segmentation(self, frame=None):
        path = "/Users/regevazran/Desktop/technion/semester i/project c/temp pic/mouse1.jpg"
        frame = cv2.imread(path)
        self.cur_frame = frame
        self.get_rectangle()
        self.cut_frame_by_wound()
        self.segment_wound()
