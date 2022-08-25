from Inferer import Inferer
import time
import torch
import cv2
import numpy as np
from mss import mss
import keyboard
import win32api
import pandas as pd
from statistics import mean

inferer = Inferer(weights="yolov6n.pt", device=0, yaml="data/dataset.yaml", img_size=640, half=False,
                  agnostic_nms=False,
                  max_det=1000, iou_thres=0.45, conf_thres=0.25)

prev_frame_time = 0
new_frame_time = 0
showFPS=True
fpsArr=[]
font = cv2.FONT_HERSHEY_SIMPLEX

classes = [0]

inferer = Inferer(weights="yolov6n.pt", device=0, yaml="data/dataset.yaml", img_size=960, half=False,
                  agnostic_nms=False,
                  max_det=1000, iou_thres=0.45, conf_thres=0.25, classes=classes)

with mss() as sct:
#   crop screenshot if needed
    monitor = {"top": 240, "left": 800, "width": 960, "height": 960}

    while True:
        screenshot = np.array(sct.grab(monitor))[:, :, :3]
        screenshot = np.ascontiguousarray(screenshot)
        img, img_src = Inferer.precess_image(screenshot, inferer.img_size, inferer.model.stride, False)
        det = inferer.infer(img, img_src)
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)
                label = None if False else (inferer.class_names[class_num] if False else f'{inferer.class_names[class_num]} {conf:.2f}')
                Inferer.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label,
                                           color=Inferer.generate_colors(class_num, True))
            print(pd. DataFrame(det.cpu().numpy()))
            image = np.array(img_src)

        if showFPS == True:
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            if len(fpsArr) <= 10:
                fpsArr.append(fps)
            else:
                fpsArr = fpsArr[1::]
                fpsArr.append(fps)
            fpsToShow = mean(fpsArr)
            cv2.putText(screenshot, str(int(fpsToShow)), (7, 70), font, 2, (100, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("frame", img_src)
        if (cv2.waitKey(1) == ord('q')):
            break
cv2.destroyAllWindows()
