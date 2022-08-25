from Inferer import Inferer
import time
import torch
import cv2
import numpy as np
from mss import mss
import keyboard
# import serial
import win32api
import pandas as pd
from statistics import mean

inferer = Inferer(weights="yolov6n.pt", device=0, yaml="data/dataset.yaml", img_size=640, half=False,
                  agnostic_nms=False,
                  max_det=1000, iou_thres=0.45, conf_thres=0.25)

enableSpray = True
FOVx = 400
FOVy = 300
FOVBoxMin = (int((960 - FOVx) / 2), int((960 - FOVy) / 2))
FOVBoxMax = (int(FOVx + (960 - FOVx) / 2), int(FOVy + (960 - FOVy) / 2))

prev_frame_time = 0
new_frame_time = 0
showFPS=True
fpsArr=[]
font = cv2.FONT_HERSHEY_SIMPLEX



def SendCordinates(code):
    code = str(code)
    print(code)
    # print(code)
    # arduino.write(str.encode(code))
    pass


def CalculateDistance(x, y, mode):  ## distance : (-100,100) -> "n,100,p,100*"
    if x < 0:
        x *= -1
        x_d = "n"
    else:
        x_d = "p"
    if y < 0:
        y *= -1
        y_d = "n"
    else:
        y_d = "p"
    x_v = int(x / 5)
    y_v = int(y / 5)
    code = x_d + "," + str(x_v) + "," + y_d + "," + str(y_v) + "," + mode + "*"
    return code

def CalculateHeadLevel(df, xmin, xmax, ymin, ymax):
    df = df.assign(headLevelX=lambda x: x.xmin + (x.xmax - x.xmin) / 2)
    df = df.assign(headLevelY=lambda x: x.ymin + (x.ymax - x.ymin) / 7)


classes = [0]

inferer = Inferer(weights="yolov6n.pt", device=0, yaml="data/dataset.yaml", img_size=960, half=False,
                  agnostic_nms=False,
                  max_det=1000, iou_thres=0.45, conf_thres=0.25, classes=classes)

with mss() as sct:
    monitor = {"top": 240, "left": 800, "width": 960, "height": 960}

    while True:
        screenshot = np.array(sct.grab(monitor))[:, :, :3]
        screenshot = np.ascontiguousarray(screenshot)
        img, img_src = Inferer.precess_image(screenshot, inferer.img_size, inferer.model.stride, False)
        det = inferer.infer(img, img_src)

        # results = inferer(screenshot, size=640)
        # df = pd.DataFrame(x.numpy())
        # df = results.pandas().xyxy[0]
        # df = df.assign(headLevelX=lambda x: x.xmin + (x.xmax - x.xmin) / 2)
        # df = df.assign(headLevelY=lambda x: x.ymin + (x.ymax - x.ymin) / 7)
        # df = df.assign(distanceFromCrosshair=lambda x: (x.headLevelX - 480) ** 2 + (x.headLevelY - 480) ** 2)
        # df = df.query("distanceFromCrosshair <= @FOV**2")
        # df = df.sort_values(by=['distanceFromCrosshair'])
        cv2.putText(screenshot, str(enableSpray), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=5)
        cv2.rectangle(screenshot, FOVBoxMin, FOVBoxMax, (0, 0, 255), 2)
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)
                label = None if False else (inferer.class_names[class_num] if False else f'{inferer.class_names[class_num]} {conf:.2f}')
                Inferer.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label,
                                           color=Inferer.generate_colors(class_num, True))
            # Important
            print(pd. DataFrame(det.cpu().numpy()))
            image = np.array(img_src)

        # try:
        #     # print(df)
        #     xmin = int(df.iloc[0, 0])
        #     ymin = int(df.iloc[0, 1])
        #     xmax = int(df.iloc[0, 2])
        #     ymax = int(df.iloc[0, 3])
        #     head_level = (int(df.iloc[0, 7]), int(df.iloc[0, 8]))
        #     # cv2.line(screenshot, (480,480), (head_level), (0,255,255), 3)
        #     cv2.circle(screenshot, head_level, 4, (0, 255, 0), thickness=-1)
        #     cv2.rectangle(screenshot, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        #     distance = (head_level[0] - 480, head_level[1] - 480)
        #
        #     if win32api.GetKeyState(0x01) < 0:
        #         if enableSpray == True:
        #             if (abs(head_level[1] - 480) <= FOVy / 2) and (abs(head_level[0] - 480) <= FOVx / 2):
        #                 code = CalculateDistance(int(distance[0]), int(distance[1]) + abs(sprayIndex) * 20, "spray")
        #                 SendCordinates(code)
        #                 time.sleep(0.025)
        #                 if sprayIndex < 3:
        #                     sprayIndex += 1
        #     else:
        #         sprayIndex = -1
        #     if keyboard.is_pressed("alt"):
        #         if (abs(head_level[1] - 480) <= FOVy / 2) and (abs(head_level[0] - 480) <= FOVx / 2):
        #             code = CalculateDistance(int(distance[0]), int(distance[1]), "trigg")
        #             SendCordinates(code)
        #             time.sleep(0.175)
        # except:
        #     print("", end="")
        # if keyboard.is_pressed("n"):
        #     if enableSpray == True:
        #         enableSpray = False
        #         time.sleep(0.175)
        #     else:
        #         enableSpray = True
        #         time.sleep(0.175)
        # new_frame_time = time.time()
        # fps = 1 / (new_frame_time - prev_frame_time)
        # prev_frame_time = new_frame_time
        # fps = str(int(fps))
        # print("FPS:"+fps)
        if showFPS == True:
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            if len(fpsArr) <= 10:
                fpsArr.append(fps)
            else:
                fpsArr = fpsArr[0:-1]
                fpsArr.append(fps)
            fpsToShow = mean(fpsArr)
            cv2.putText(screenshot, str(int(fpsToShow)), (7, 70), font, 2, (100, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("frame", img_src)
        if (cv2.waitKey(1) == ord('q')):
            break
cv2.destroyAllWindows()

video_path = "inputvideo.mp4"
# classes=[0,1,2,3]
#
# inferer = Inferer(weights="yolov6n.pt", device=0, yaml="data/dataset.yaml", img_size=960, half=False, agnostic_nms=False,
#                   max_det=1000, iou_thres=0.45, conf_thres=0.25,classes=classes)


# video  =cv2.VideoCapture(video_path)
# ret,img_src = video.read()
# output = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'DIVX'),30,(img_src.shape[1],img_src.shape[0]))
#
# while True:
#     if ret:
#         # start = time.time()
#         print(img_src.shape)
#         img, img_src = Inferer.precess_image(img_src,inferer.img_size,inferer.model.stride,False)
#         print(img_src.shape)
#         det = inferer.infer(img,img_src)
#
#         # end = time.time() - start
#         for *xyxy, conf, cls in reversed(det):
#             class_num = int(cls)
#             label = None if False else (inferer.class_names[class_num] if False else f'{inferer.class_names[class_num]} {conf:.2f}')
#             Inferer.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))
#         image=np.array(img_src)
#         output.write(image)
#         cv2.imshow("frame",image)
#
#         ret,img_src = video.read()
#     else:
#         break
#
# output.release()
# video.release()
