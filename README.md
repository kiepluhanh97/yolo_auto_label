# yolo_auto_label
This repo to help you create txts file from images. Using for traning object detection from pretrain model that you had. It will save your time to label imgs.

# Download pretrain model
https://drive.google.com/file/d/1laNMDk-eSf1eb0ewv29Ojzuy-54zGN6Q/view?usp=sharing

Copy to v4 folder and run:
# Run mode GPU with darknet
python3 create_anno_gpu.py --model-path 'v4/yolov4.weights' --cfg-path 'v4/yolov4.cfg' --meta-path 'v4/yolov4.data' --img-folder 'images' --txt-folder 'txts'
Note: You must install nvidia cuda. Darknet lib reference from https://github.com/AlexeyAB/darknet
    - All test on RTX 2080 Ti, cuda 10.1. If you are using another version, maybe you have to rebuild darknet to get libdarknet.so file.

# Run CPU mode with opencv
python3 create_anno_cpu.py --model-path 'v4/yolov4.weights' --cfg-path 'v4/yolov4.cfg' --meta-path 'v4/yolov4.data' --img-folder 'images' --txt-folder 'txts'

Pretrained model that I trained is on custom dataset only have box for person and head. If you have any question, feel free to ask me