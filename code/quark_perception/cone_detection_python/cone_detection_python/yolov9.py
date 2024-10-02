import os
import sys
from pathlib import Path
import time

import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, letterbox
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class Yolov9:
    def __init__(self, 
                weights,  # model path or triton URL                 
                classes = None,
                data = 'C:/Users/seren/OneDrive/Área de Trabalho/TCC/signal_detector/data.yaml',
                img_size=1280,  # inference size (height, width)
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.45,  # NMS IOU threshold
                ) -> None:
        
        
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.classes = classes
        self.data = data
        self.startup()

    def startup(self):
        # Initialize
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = True

        # Load model
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=self.half)
        
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.img_size, s=self.stride)  # check image size
        


       
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        t0 = time.time()

    def detect(self, imgs):
        # Run inference
        img0 = imgs.copy()

        # Letterbox
        img = [letterbox(x, self.img_size, auto=True, stride=self.stride)[0] for x in img0]
        print(f'Shape img: {np.shape(img)}, img0: {np.shape(img0)}')
        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0    
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            # Inference
        
        pred = self.model(img, augment=False)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, False, max_det=1000)

        centers = []
        # Process predictions
        for i, det in enumerate(pred):  # per image

            if self.webcam:  # batch_size >= 1
                
                s, im0 = '%g: ' % i, img0[i].copy()
            else:
                s, im0 = '', img0
            
            annotator = Annotator(im0, line_width=3, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    centers.append((xyxy, conf))
                    c = int(cls)  # integer class
                    label =  f'{self.names[c]} \n {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                print(conf)
        return im0, centers
