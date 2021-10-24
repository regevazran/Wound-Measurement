import torch
import cv2
import numpy as np
from yolo.yolov5.detect import run
import tkinter.filedialog
import os

# !python detect.py --weights runs/train/yolov5s_results2/weights/yolov5_weights.pt --img 416 --conf 0.4 --source "/content/drive/MyDrive/project_c/valuationdata"


def show_image(detection, image_copy):
    cv2.rectangle(image_copy, pt1=(detection['x0'], detection['y0']),
                  pt2=(detection['x1'], detection['y1']), color=(255, 0, 0), thickness=3)
    cv2.putText(image_copy, text=str(detection['class'] + str(detection['confidence'])),
                org=(detection['x0'], detection['y0'] - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, color=(255, 0, 0), thickness=3)
    img = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))
    cv2.imshow("img", img)


def yolo_results_to_dict(res):
    rectangle = np.asarray(res.xyxy[0])[0]
    detection_dict = {'x0': int(rectangle[0]), 'y0': int(rectangle[1]), 'x1': int(rectangle[2]),
                      'y1': int(rectangle[3]), 'confidence': round(rectangle[4], 3),
                      'class': 'wound' if rectangle[5] == 0 else 'None'}
    return detection_dict


def test_yolov5():
    images = tkinter.filedialog.askopenfiles()

    src_directory = 'C:/Users/tomer/PycharmProjects/Wound-Measurement/yolo/yolov5/'

    model = torch.hub.load(src_directory, 'custom', path=str(src_directory + 'yolov5_weights.pt'), source='local')

    for file in images:
        img = cv2.imread(file.name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detection_dict = yolo_results_to_dict(model(img))
        show_image(img.copy(), detection_dict)
        cv2.waitKey(0)


class YoloAlgo:
    def __init__(self):
        self.src_directory = str(os.getcwd() + "/yolo/yolov5/")
        self.weights_path = self.src_directory + "yolov5_weights.pt"
        self.model = torch.hub.load(self.src_directory, 'custom', path=str(self.src_directory + 'yolov5_weights.pt'), source='local')

    def run(self, image):
        image_copy = image.copy()
        detection_dict = yolo_results_to_dict(self.model(image))
        show_image(detection_dict, image_copy)
        return detection_dict



