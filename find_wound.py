import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from tqdm import tqdm
import matplotlib.pyplot as plt
coordinates = [0.64, 0.521, 0.088, 0.168]
# coordinates = [0.520399, 0.392470, 0.190104, 0.074436]
# coordinates = [0.514974, 0.518012, 0.037326, 0.072483]
mouse_path = "C:/mouse_original.jpg"
# mouse_path = "C:/mouse_original_2.jpg"
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


if __name__ == '__main__':
	mouse_original = cv2.imread(mouse_path)
	mouse_original_YCR_CB = cv2.cvtColor(mouse_original, cv2.COLOR_BGR2HSV)

	# mouse_gray = cv2.cvtColor(mouse_original, cv2.COLOR_BGR2GRAY)
	rect = get_rect(mouse_original_YCR_CB)

	only_wound = mouse_original_YCR_CB[-50 + rect[1]:100 + rect[1] + rect[3], -50 + rect[0]:100 + rect[0] + rect[2]]
	only_wound_original = mouse_original[-50 + rect[1]:100 + rect[1] + rect[3], -50 + rect[0]:100 + rect[0] + rect[2]]

	cv2.imshow("only_wound", only_wound)
	cv2.waitKey(0)

	init = get_init_for_snake(only_wound)

	snake = active_contour(gaussian(only_wound, 21), init)

	last_size = 100000
	converged = 0
	diff_thresh = 0.005
	iterations = 500

	for i in range(0, iterations):
		snake = active_contour(gaussian(only_wound, 7), snake, boundary_condition='periodic')
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
