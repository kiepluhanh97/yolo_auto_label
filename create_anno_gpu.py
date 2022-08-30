import numpy as np
import cv2
from imutils import paths
import os
from utils import darknet
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
   network, class_names, class_colors = darknet.load_network(
      CONFIG_PATH,
      META_PATH,
      MODEL_PATH,
      batch_size=1
   )
   images_folder = args.img_folder
   txts_folder = args.txt_folder
   if not os.path.exists(txts_folder):
      os.mkdir(txts_folder)


   number_img = 0
   for eachFile in listdir(images_folder):
      if eachFile[len(eachFile) - 3:] == 'jpg':
         number_img += 1
         print(eachFile)
         fileName = eachFile[:len(eachFile) - 4]
         img = cv2.imread(images_folder + '/' + eachFile)
         img = cv2.resize(img, (416,416))
         (H, W) = img.shape[:2]
         darknet_image = darknet.make_image(W,H,3)
         darknet.copy_image_from_bytes(darknet_image, img.tobytes())
         detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.3)
         darknet.free_image(darknet_image)
         stringOutput = ''
         for detection in detections:
               # print(detection)
               cls = class_names.index(detection[0])
               stringOutput += str(cls) + ' ' + str(detection[2][0]/W) + ' ' + str(detection[2][1]/H) + ' ' + str(detection[2][2]/W) + ' ' + str(detection[2][3]/H) + '\n'
         f = open(txts_folder + '/' + fileName + '.txt', 'w')
         f.write(stringOutput)
         f.close()
