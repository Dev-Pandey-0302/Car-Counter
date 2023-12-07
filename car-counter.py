import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


cap = cv2.VideoCapture("../Car_Counter/videos/vid.mp4")
cap.set(3, 1280)
cap.set(4, 720)

model= YOLO("../YOLO-WEIGHTS/yolov8l.pt")
mask=cv2.imread("mask_2.png")
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
    "dining table", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


limits=[0,170,320,170]


total_count=[]

while True:
    success, img = cap.read()
    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)

    imgRegion=cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)

    detections=np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w,h=x2-x1,y2-y1



            # Confidence Score
            conf = math.ceil((box.conf[0]*100))/100




            # Class Name
            cls=box.cls[0]
            classType=classNames[int(cls)]

            if (classType=="car" or classType=="bus" or classType=="truck" or classType=="bicycle") and conf>=0.4:
                # Bounding box
                # cvzone.cornerRect(img, (x1, y1, w, h), l=5, t=2, rt=5)

                # Showing class + confidence
                    #cvzone.putTextRect(img, f'{classType} {conf}', (max(0, x1), max(35, y1)),scale=1, thickness=1, offset=1)

                #tracking

                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))



    results_Tracker=tracker.update(detections)

    #Drawing line
    cv2.line(img, (limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

    for result in results_Tracker:
        x1,y1,x2,y2,id=result
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=5, t=2, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=1)

        cx,cy= x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)


        if ((limits[0]<cx<limits[2]) and (limits[1]-10<cy<limits[3]+10)):
            if total_count.count(id) ==0:
                total_count.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f'Count: {len(total_count)}', (50, 50))


    cv2.imshow("Image_test", img)

    cv2.waitKey(1)