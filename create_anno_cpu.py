import numpy as np
import cv2
from imutils import paths
import os
from os import listdir
import argparse


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--model-path', help='link to ai model', type=str, default="", required=True)
   parser.add_argument('--cfg-path', help='link to cfg file', type=str, default="", required=True)
   parser.add_argument('--meta-path', help='link to meta file', type=str, default="", required=True)
   parser.add_argument('--img-folder', help='link to image folder', type=str, default="", required=True)
   parser.add_argument('--txt-folder', help='link to txt folder', type=str, default="", required=True)

   args = None
   try:
      args = parser.parse_args()
   except:
      print("==== Parse argument exception")
   print(args)
#  Namespace(cfg_path='v4/yolov4.cfg', meta_path='v4/yolov4.data', model_path='v4/yolov4.weights')
   MODEL_PATH = args.model_path
   CONFIG_PATH  = args.cfg_path
   META_PATH = args.meta_path
   net = cv2.dnn.readNet(MODEL_PATH, CONFIG_PATH)
   model = cv2.dnn_DetectionModel(net)
   model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
   images_folder = args.img_folder
   txts_folder = args.txt_folder
   if not os.path.exists(txts_folder):
      os.mkdir(txts_folder)

   CONFIDENCE_THRESHOLD = 0.5
   NMS_THRESHOLD = 0.1
   number_img = 0
   for eachFile in listdir(images_folder):
      if eachFile[len(eachFile) - 3:] == 'jpg':
         number_img += 1
         print(eachFile)
         fileName = eachFile[:len(eachFile) - 4]
         img = cv2.imread(images_folder + '/' + eachFile)
         img = cv2.resize(img, (416,416))
         (H, W) = img.shape[:2]
         classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
         stringOutput = ''
         for cls, box in zip(classes, boxes):
            cx, cy = box[0] + box[2]/2, box[1] + box[3]/2
            stringOutput += str(cls[0]) + ' ' + str(cx/W) + ' ' + str(cy/H) + ' ' + str(box[2]/W) + ' ' + str(box[3]/H) + '\n'
         f = open(txts_folder + '/' + fileName + '.txt', 'w')
         f.write(stringOutput)
         f.close()
