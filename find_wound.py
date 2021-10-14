import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour

import matplotlib.pyplot as plt
from skimage import data
from skimage.exposure import histogram


coordinates1 = [0.612305,0.677083,0.065755,0.120660]
coordinates2 = [0.551975, 0.507161, 0.027127, 0.072483]
coordinates3 = [0.435547, 0.490126, 0.132378, 0.052734]
coordinates4 = [0.638563, 0.494792, 0.056641, 0.151910]
coordinates5 = [0.667101, 0.531467, 0.065104, 0.127170]
coordinates6 = [0.496094, 0.420573, 0.098090, 0.027344]
mouse_path1 = "/Users/regevazran/Desktop/technion/semester i/project c/temp pic/mouse1.jpg"
mouse_path2 = "/Users/regevazran/Desktop/technion/semester i/project c/temp pic/mouse2.jpg"
mouse_path3 = "/Users/regevazran/Desktop/technion/semester i/project c/temp pic/mouse3.jpg"
mouse_path4 = "/Users/regevazran/Desktop/technion/semester i/project c/temp pic/mouse4.jpg"
mouse_path5 = "/Users/regevazran/Desktop/technion/semester i/project c/temp pic/mouse5.jpg"
mouse_path6 = "/Users/regevazran/Desktop/technion/semester i/project c/temp pic/mouse6.jpg"


mouse_path = mouse_path6
coordinates = coordinates6

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
        snake = active_contour(gaussian(gaussianed_wound, 7), snake)
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

def grab_cut(img, rect):
    # cut a square from the picture around the relevant area

    img = img[-150 + rect[0][1]:140 + rect[1][1], -200 + rect[0][0]:230 + rect[1][0]]
    # img = cv2.GaussianBlur(img, (9, 9), 0)
    rect = [(120,120),(img.shape[1]-120,img.shape[0]-120)]
    rect_for_grab_cut = [120,120,rect[1][0],rect[1][1]]

    img_original = img.copy()

    # set mask for the grabCut to update
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # use grabCut to mask the background from the segmented objects
    cv2.grabCut(img,mask,rect_for_grab_cut,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    # contours = [np.array([p0,p1,p2,p3], dtype=np.int32)]
    # cv2.rectangle(img_original, rect[0], rect[1], (255, 0, 255), 2)

    cv2.imshow("original",img_original)
    cv2.waitKey(1)
    cv2.imshow("grab cut",img)
    cv2.waitKey(0)
    return  img

if __name__ == '__main__':
    # Get Wound
    mouse_original = cv2.imread(mouse_path)
    rect = get_rect(mouse_original)
    only_wound_original = mouse_original[-50 + rect[1]:100 + rect[1] + rect[3], -50 + rect[0]:100 + rect[0] + rect[2]]

    # Semantic segmentation pascalvoc model
    # pascalvoc_model()

    # grab cat
    img_grab_cut = grab_cut(mouse_original, rect)
    # only_wound_grab_cut = img_grab_cut[-50 + rect[1]:100 + rect[1] + rect[3], -50 + rect[0]:100 + rect[0] + rect[2]]

    # Semantic segmentation snake (active contour) algorithm
    # snake_algorithm(only_wound_original, mouse_original)

    # Semantic Segmentation using canny + search for optimal value
    # canny_with_trackbar()

    # chan vese


