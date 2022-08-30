import numpy as np
import cv2
from imutils import paths
import os
import darknet
#################### CONFIG ############################
IMG_DIR    = './data/imgs'
ANNS_DIR   = './data/anns'
MODEL_PATH = './cfg/custom-obj_26000.weights'
# MODEL_PATH = '/media/iot/phaptq/darknet_bsx_text_update/backup/custom-obj.backup'
########################################################
if __name__ == '__main__':
    configPath  = 'cfg/yolov3.cfg'
    metaPath = "cfg/yolo.data"
    labels = open('./cfg/yolo.names').read().strip().split('\n')
    netMain = darknet.load_net_custom(configPath.encode("ascii"), MODEL_PATH.encode("ascii"), 0, 1)  # batch size = 1
    metaMain = darknet.load_meta(metaPath.encode("ascii"))

    input_video = cv2.VideoCapture("video04.mp4")
    total = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video = None
    W = None
    H = None
    frame_number = 0
    while True:
        frame_number += 1
        print("Writing frame {}/{}".format(frame_number,total))
        ret, frame = input_video.read()
        (H, W) = frame.shape[:2]
        darknet_image = darknet.make_image(W,H,3)
        darknet.copy_image_from_bytes(darknet_image, frame.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.3)
        darknet.free_image(darknet_image)
        boxes = []
        confidences = []
        classIDs = []
        for detection in detections:
            box_2 = [int(detection[2][0] - detection[2][2] / 2),int(detection[2][1] - detection[2][3] / 2),detection[2][2],detection[2][3]]
            boxes.append(box_2)
            confidences.append(float(detection[1]))
            classIDs.append(detection[0])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
        for i in indices:
            i = i[0]
            box = boxes[i]
            posObj = classIDs[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            x1,y1,x2,y2 = round(x), round(y), round(x+w), round(y+h)
            cv2.putText(frame, labels[posObj], (x1, y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame,str(confidences[i]) , (x1, y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            print(x,y,w,h,posObj)
        if output_video is None:
            fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
            output_video = cv2.VideoWriter('output4.avi', fourcc, 30, (W, H), True)
        output_video.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    input_video.release()
    cv2.destroyAllWindows()
