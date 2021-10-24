import torch
import cv2
import numpy as np
import os
from yolo.yolov5.detect import run
import tkinter.filedialog


# !python detect.py --weights runs/train/yolov5s_results2/weights/best.pt --img 416 --conf 0.4 --source "/content/drive/MyDrive/project_c/valuationdata"


def test_yolov5():
    imgs = tkinter.filedialog.askopenfiles()
    # os.chdir('C:/Users/tomer/PycharmProjects/Wound-Measurement/yolo/yolov5')
    src_directory = 'C:/Users/tomer/PycharmProjects/Wound-Measurement/yolo/yolov5/'
    # weights_path = 'C:/Users/tomer/PycharmProjects/Wound-Measurement/yolo/yolov5/best.pt'

    model = torch.hub.load('C:/Users/tomer/PycharmProjects/Wound-Measurement/yolo/yolov5/', 'custom', path=str('C:/Users/tomer/PycharmProjects/Wound-Measurement/yolo/yolov5/' + 'best.pt'), source='local')
    for file in imgs:
        img = cv2.imread(file.name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)
        # print("found detection on:\n", results.pandas().xyxy[0])  # img1 predictions (pandas)ct[0], rect[1], (0, 0, 255), 3)
        rectangle = np.asarray(results.xyxy[0])[0]
        detection_dict = {'x0': int(rectangle[0]), 'y0': int(rectangle[1]), 'x1': int(rectangle[2]), 'y1': int(rectangle[3]), 'confidence': round(rectangle[4], 3), 'class': 'wound' if rectangle[5] == 0 else 'None'}
        print(detection_dict)
        cv2.rectangle(img, pt1=(detection_dict['x0'], detection_dict['y0']), pt2=(detection_dict['x1'], detection_dict['y1']), color=(255, 0, 0), thickness=3)

        cv2.putText(img, text=str(detection_dict['class'] + str(detection_dict['confidence'])), org=(detection_dict['x0'], detection_dict['y0'] - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 0, 0),thickness=3 )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))
        cv2.imshow("img", img)
        cv2.waitKey(0)


def run_yolov5(weights_path, imgs):
    run(weights=weights_path, imgsz=640, source=imgs, conf_thres=0.25, view_img=False)
