import math

import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from PIL import Image


import matplotlib.pyplot as plt
from skimage import data
from skimage.exposure import histogram


coordinates1 = [0.612305,0.677083,0.065755,0.120660]
coordinates2 = [0.551975, 0.507161, 0.027127, 0.072483]
coordinates3 = [0.435547, 0.490126, 0.132378, 0.052734]
coordinates4 = [0.638563, 0.494792, 0.056641, 0.151910]
coordinates5 = [0.667101, 0.531467, 0.065104, 0.127170]
coordinates6 = [0.496094, 0.420573, 0.098090, 0.027344]
mouse_path1 = "/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/mouse1.jpg"
mouse_path2 = "/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/mouse2.jpg"
mouse_path3 = "/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/mouse3.jpg"
mouse_path4 = "/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/mouse4.jpg"
mouse_path5 = "/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/mouse5.jpg"
mouse_path6 = "/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/mouse6.jpg"


mouse_path = mouse_path3
coordinates = coordinates3


def histogram_equalization(self, image):
    # convert from RGB color-space to YCrCb
    image = np.array(image)
    image = cv2.resize(image, (int(image.shape[1] * 0.4), int(image.shape[0] * 0.4)))
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    # cv2.imshow('img no HE', image)
    # cv2.imshow('equalized_img', equalized_img)
    # key = cv2.waitKey(0)
    return equalized_img
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
def preprocess_pic(self, pic):
        pic = np.array(pic)
        cv2.imshow("original pic", cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
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
def get_wound_contour(self, pic):
        blur = cv2.GaussianBlur(pic, (5, 5), 0)
        edges = cv2.Canny(blur, 20, 70 )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("edges", edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours
def deep_segmentation(self,pic):
        path = '/Users/regevazran1/Desktop/technion/semester i/project c/temp pic/'
        result = cv2.imwrite(path+'temp_pic.jpg', pic)
        from pixellib.semantic import semantic_segmentation

        segment_image = semantic_segmentation()
        segment_image.load_pascalvoc_model(path+"deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
        segment_image.segmentAsPascalvoc(path+'temp_pic.jpg', output_image_name=path+"image_new.jpg")
        return
def spectral_clustering(self, img):
        mask = img.astype(bool)
        img = img.astype(float)

        img += 1 + 0.2 * np.random.randn(*img.shape)
        print("start image to graph")

        graph = Image.img_to_graph(img, mask=mask)
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
def get_contours(self, pic):
        blur = cv2.GaussianBlur(pic, (5, 5), 0)
        _, thresh_pic = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow("thresh", thresh_pic)
        contours, _ = cv2.findContours(thresh_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
def pic2HSV(self, pic):
        pic = np.array(pic)
        pic = cv2.resize(pic, (int(pic.shape[1]*0.4), int(pic.shape[0]*0.4)))
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
def get_rect(mouse_img):
    shape = mouse_img.shape
    width = int(coordinates[2] * shape[1])
    height = int(coordinates[3] * shape[0])
    top_left_x = int(coordinates[0] * shape[1] - width / 2)
    top_left_y = int(coordinates[1] * shape[0] - height / 2)
    return top_left_x, top_left_y, width, height
def get_init_for_snake(only_wound_img,wound_rect):

    center_x = int(only_wound_img.shape[0] / 2)
    center_y = int(only_wound_img.shape[1] / 2)
    rect_h = wound_rect[1][1] - wound_rect[0][1]
    rect_w = wound_rect[1][0] - wound_rect[0][0]
    radius = int(max(rect_h,rect_w)/2)+50
    s = np.linspace(0, 2 * np.pi, 400)
    r = center_y + radius * np.sin(s)
    c = center_x + radius * np.cos(s)
    init = np.array([r, c]).T
    return init
def pascalvoc_model():
    pass
    # segment_image = semantic_segmentation()
    # segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    # segment_image.segmentAsPascalvoc(mouse_path, output_image_name="mouse_segmented.jpg")
    # exit(-1)
def snake_algorithm(only_wound_original,wound_rect):
    init = get_init_for_snake(only_wound_original,wound_rect)

    # Pre Processing
    # mouse_gray = cv2.cvtColor(mouse_original.copy(), cv2.COLOR_BGR2GRAY)
    # mouse_normalized = cv2.equalizeHist(mouse_gray)

    # mouse_original_YCR_CB = cv2.cvtColor(mouse_original, cv2.COLOR_BGR2HSV)
    only_wound = only_wound_original.copy()

    gaussianed_wound = cv2.GaussianBlur(only_wound, (9, 9), 0)


    snake = active_contour(gaussian(gaussianed_wound, 3), init)
    init = np.array(init, dtype=np.int32)
    snake_contour = np.array(snake, dtype=np.int32)
    tmp_frame = only_wound_original.copy()

    cv2.drawContours(tmp_frame, [init], -1, (255, 0, 0), 2)
    cv2.drawContours(tmp_frame, [snake_contour], -1, (0, 0, 255), 2)
    cv2.imshow("mouse_original", tmp_frame)
    cv2.waitKey(0)

    last_size = 100000
    converged = 0
    diff_thresh = 0.005
    iterations = 500

    for i in range(0, iterations):
        snake = active_contour(gaussian(gaussianed_wound, 3), snake)
        snake_contour = np.array(snake, dtype=np.int32)
        cur_snake_contour_size = cv2.contourArea(snake_contour)
        contour_area_change_ratio = 100 * (abs(cur_snake_contour_size - last_size) / last_size)
        converged = converged + 1 if contour_area_change_ratio < diff_thresh else 0
        print("iteration:", i, round(contour_area_change_ratio, 3), "%. counter: ", converged)
        if converged == 3:
            break
        last_size = cur_snake_contour_size
        tmp_frame = only_wound_original.copy()
        cv2.drawContours(tmp_frame, [init], -1, (255, 0, 0), 2)
        cv2.drawContours(tmp_frame, [snake_contour], -1, (0, 0, 255), 2)
        cv2.imshow("mouse_original", tmp_frame)
        cv2.waitKey(1)

    contours = [np.array(snake, dtype=np.int32)]
    cv2.drawContours(only_wound_original, contours, -1, (0, 0, 255), 2)
    cv2.imshow("mouse_original", only_wound_original)
    cv2.waitKey(0)
def canny_with_trackbar(only_wound_original):
    src = only_wound_original
    max_lowThreshold = 100
    window_name = 'Edge Map'
    title_trackbar = 'Min Threshold:'
    ratio = 3
    kernel_size = 3

    def CannyThreshold(val):
        low_threshold = val
        img_blur = cv2.blur(src_gray, (3, 3))
        detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
        mask = detected_edges != 0
        dst = src * (mask[:, :, None].astype(src.dtype))
        cv2.imshow(window_name, dst)

    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow(window_name)
    cv2.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)
    CannyThreshold(0)
    cv2.waitKey()
# chan vese
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
def chan_vese(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image)
    image = img_as_float(image)
    cv = chan_vese(image, mu=0.01, lambda1=1, lambda2=1, tol=1e-3, max_iter=500, dt=0.5, init_level_set="checkerboard",
                   extended_output=True)
    image = np.where(cv[0],0.9,0)
    print(image.shape)
    print(cv[1])
    cv2.imshow("chan vese", image)
    cv2.waitKey(0)
# segmenting mouse wound using grabCut
import matplotlib.patches as patches
def preprocess_wound(img_original):
    # Read the image and perfrom an OTSU threshold
    img = img_original.copy()
    kernel = np.ones((3, 3), np.uint8)

    # Perform closing to remove hair and blur the image
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    blur = cv2.blur(closing, (15, 15))

    # Binarize the image
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("img",img)
    cv2.imshow("closing", closing)
    cv2.imshow("blur", blur)
    cv2.imshow("thresh", thresh)
    return img

def find_squares(img):

    return

if __name__ == '__main__':
    # Get Wound
    mouse_original = cv2.imread(mouse_path)
    rect = get_rect(mouse_original)
    only_wound_original = mouse_original[-100 + rect[1]:150 + rect[1] + rect[3], -100 + rect[0]:150 + rect[0] + rect[2]]
    rect = [(100,100),(only_wound_original.shape[1]-100,only_wound_original.shape[0]-100)]
    # Semantic segmentation pascalvoc model
    # pascalvoc_model()
    # preprocess img
    # only_wound_preprocess = preprocess_wound(only_wound_original)
    # Semantic segmentation snake (active contour) algorithm
    # snake_algorithm(only_wound_original)

    # Semantic Segmentation using canny + search for optimal value
    # canny_with_trackbar()

    # chan vese


