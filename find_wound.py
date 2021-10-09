import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour

import matplotlib.pyplot as plt
from skimage import data
from skimage.exposure import histogram


# coordinates = [0.64, 0.521, 0.088, 0.168]
coordinates = [0.520399, 0.392470, 0.190104, 0.074436]
# coordinates = [0.514974, 0.518012, 0.037326, 0.072483]
# mouse_path = "C:/mouse_original.jpg"
mouse_path = "C:/mouse_original_2.jpg"
# mouse_path = "C:/mouse_original_3.jpg"


def get_rect(mouse_img):
	shape = mouse_img.shape
	width = int(coordinates[2] * shape[1])
	height = int(coordinates[3] * shape[0])
	top_left_x = int(coordinates[0] * shape[1] - width / 2)
	top_left_y = int(coordinates[1] * shape[0] - height / 2)
	return top_left_x, top_left_y, width, height


def get_init_for_snake(only_wound_img):
	center_x = int(only_wound_img.shape[0] / 2)
	center_y = int(only_wound_img.shape[1] / 2)
	radius = min(int(center_y), int(center_x)) - 20

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


def snake_algorithm(only_wound_original, mouse_original):
	init = get_init_for_snake(only_wound_original)

	# Pre Processing
	# mouse_gray = cv2.cvtColor(mouse_original.copy(), cv2.COLOR_BGR2GRAY)
	# mouse_normalized = cv2.equalizeHist(mouse_gray)
	mouse_gaussianed = cv2.GaussianBlur(mouse_original, (13, 13), 0)
	# mouse_original_YCR_CB = cv2.cvtColor(mouse_original, cv2.COLOR_BGR2HSV)
	pre_processed_wound = mouse_gaussianed[-50 + rect[1]:100 + rect[1] + rect[3], -50 + rect[0]:100 + rect[0] + rect[2]]

	cv2.imshow("pre processed wound", pre_processed_wound)
	cv2.waitKey(1)

	snake = active_contour(gaussian(pre_processed_wound, 7), init)

	last_size = 100000
	converged = 0
	diff_thresh = 0.005
	iterations = 500

	for i in range(0, iterations):
		snake = active_contour(gaussian(pre_processed_wound, 3), snake, boundary_condition='periodic')
		snake_contour = np.array(snake, dtype=np.int32)
		cur_snake_contour_size = cv2.contourArea(snake_contour)
		contour_area_change_ratio = 100 * (abs(cur_snake_contour_size - last_size) / last_size)
		converged = converged + 1 if contour_area_change_ratio < diff_thresh else 0
		print("iteration:", i, round(contour_area_change_ratio, 3), "%. counter: ", converged)
		if converged == 3:
			break
		last_size = cur_snake_contour_size
		tmp_frame = only_wound_original.copy()
		cv2.drawContours(tmp_frame, [snake_contour], -1, (0, 0, 255), 2)
		cv2.imshow("mouse_original", tmp_frame)
		cv2.waitKey(1)

	contours = [np.array(snake, dtype=np.int32)]

	cv2.drawContours(only_wound_original, contours, -1, (0, 0, 255), 2)
	cv2.imshow("mouse_original", only_wound_original)
	cv2.waitKey(0)


def canny_with_trackbar():
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


if __name__ == '__main__':
	# Get Wound
	mouse_original = cv2.imread(mouse_path)
	rect = get_rect(mouse_original)
	only_wound_original = mouse_original[-50 + rect[1]:100 + rect[1] + rect[3], -50 + rect[0]:100 + rect[0] + rect[2]]

	# Semantic segmentation pascalvoc model
	# pascalvoc_model()

	# Semantic segmentation snake (active contour) algorithm
	# snake_algorithm(only_wound_original, mouse_original)

	# Semantic Segmentation using canny + search for optimal value
	canny_with_trackbar()
