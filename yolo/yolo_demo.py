import cv2
import numpy as np


whT = 608  # (width height Target) define the size of the input pictures to the net
confThreshold = 0.01
nmsThreshold = 0.3  # non maximum suppression threshold (low value = high suppression)
# define the class names
classesFile = 'yolo/coco.names'
classNames = []
# with open(classesFile,'r') as f:
#     classNames = f.read().rstrip('\n').split('\n')
classNames.append('wound')

# define net
modelConfiguration = 'yolo/yolov3.cfg'
modelWeights = 'yolo/yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape  # height width channels
    bbox = []  # bounding box (x,y,w,h)
    classIds = []
    confs = []  # confides values
    for output in outputs:
        for det in output:     # det = detection
            scores = det[5:]  # remove the first 5 values from a detection (x,y,w,h,confidence)
            classId = np.argmax(scores)  # get the index of the class with the highest confidence level
            confidence = scores[classId]
            if confidence > confThreshold:
                if det[np.argmax(det[0:5])] > 1: continue
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int(det[0]*wT - w/2), int(det[1]*hT - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print("num of bounding boxes with high confidence found: ",len(bbox))
    indices_to_keep = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold) # non maximum suppression: removes over lapping bboxes
                                                            # (leaves only the one with the highest confidence level
    x, y, w, h = 0, 0, 0, 0
    for i in indices_to_keep:
        i = i[0]  # because there is an extra [] in the original indices list
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return [[x, y], [x+w, y+h]]


def yolo_demo(img_original):
    # get image form web camera
    # success, img = cap.read()
    img = img_original.copy()

    # convert imgae to blob (format for the YOLO net)
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    unconnectedOutLayers = net.getUnconnectedOutLayers()  # take the three output layers
    outputNames = []
    for layerNum in unconnectedOutLayers: outputNames.append(layerNames[layerNum[0]-1])
    outputs = net.forward(outputNames)
    # each output in the list (we took three outputs) is an ndarry with a shape of (number of bounding boxes, 85)
    # the first 5 values of the 85 represents values of box (center x, center y, width, height, confidence level)
    # the other 80 are the confidence level for each class

    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    return findObjects(outputs, img)

def test_yolo():
    import tkinter.filedialog
    imgs = tkinter.filedialog.askopenfiles()
    for file in imgs:
        img = cv2.imread(file.name)
        rect = yolo_demo(img)
        cv2.rectangle(img, rect[0], rect[1], (0, 0, 255), 3)
        print(rect)
        img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))
        cv2.imshow("img", img)
        cv2.waitKey(0)
