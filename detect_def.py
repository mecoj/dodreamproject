# 무단 횡단자 찾아내는 로직
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from polytestest import hough_line_segments as hl


def project_draw(det,save_txt,gn,save_conf,txt_path,save_img,save_crop,view_img,hide_labels,names,hide_conf,im0,annotator,imc,save_dir,drivable_list, line_list):  # 1
    for *xyxy, conf, cls in reversed(det):
        if save_txt:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(f'{txt_path}.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
        if save_img or save_crop or view_img:  # Add bbox to image
            c = int(cls)  # integer class
            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            rg_x1 = xyxy[0]
            rg_y1 = xyxy[1]
            rg_x2 = xyxy[2]
            rg_y2 = xyxy[3]
            rg_center_dotX = int(rg_x1 + (rg_x2 - rg_x1) / 2)  # 모든 탐지 객체의 중심 영역 찾기위한 조건 x중심값
            rg_center_dotY = int(rg_y1 + (rg_y2 - rg_y1) / 2)  # 모든 탐지 객체의 중심 영역 찾기위한 조건 y중심값
            if int(cls) in [1, 2, 4, 6, 7]:  # 1,2,4,6,7 클래스만 탐지하는 조건
                if int(im0.shape[1] * 0.1) < rg_center_dotX and int(
                        im0.shape[1] * 0.9) > rg_center_dotX and 0 < rg_center_dotY and int(
                    im0.shape[0] * 0.2) > rg_center_dotY:  # 관심영역 안쪽만 탐지후 그림
                    annotator.box_label(xyxy, label, color=colors(c, True))

            else:
                annotator.box_label(xyxy, label, color=colors(c, True))
        if save_crop:
            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

    warning=''#warning경고문을 담기위한 변수

    red_bool = False

    person_list = []
    for *xyxy, conf, cls in reversed(det):
        x1 = xyxy[0]
        y1 = xyxy[1]
        x2 = xyxy[2]
        y2 = xyxy[3]
        if int(cls) == 5:  # 5 == 사람
            center_dotX = int(x1 + (x2 - x1) / 2)
            center_dotY = int(y2)

            cv2.line(im0, (center_dotX, center_dotY), (center_dotX, center_dotY), (255, 0, 255),10)
            person_list.append([center_dotX, center_dotY])

        if int(cls) in [1, 2, 4, 6, 7]:  # 1,2,4,6,7  == 신호등 클래스
            rg_center_dotX = int(x1 + (x2 - x1) / 2)
            rg_center_dotY = int(y1 + (y2 - y1) / 2)

            if int(im0.shape[1] * 0.1) < rg_center_dotX and \
                    int(im0.shape[1] * 0.9) > rg_center_dotX and \
                    0 < rg_center_dotY and \
                    int(im0.shape[0] * 0.2) > rg_center_dotY:
                if int(cls) == 6:
                    red_bool = True
                elif int(cls) == 1:
                    red_bool = False


    white_line_list = np.zeros_like(im0)
    for i in line_list:
        white_line_list[i[1]][i[0]]=255

    line_img, bd_img = hl(white_line_list)

    for pl in person_list:
        if pl in drivable_list and red_bool == False:
            warning='Warning'
        else :
            if bd_img[pl[1]][pl[0]] == 179 and red_bool ==False:
                warning = 'Warning'

    cv2.imwrite('./result_img/lane_img/0.jpg', line_img)  # 라인 이미지 저장 1
    cv2.imwrite('./result_img/lane_inside_img/0.jpg', bd_img)  # 라인 안쪽 검사 영역 이미지 저장 1
    # Stream results
    im0 = annotator.result()

    print("im0.shape : ", im0.shape)

    cv2.rectangle(im0, (int(im0.shape[1] * 0.1), int(0)), (int(im0.shape[1] * 0.9), int(im0.shape[0] * 0.2)),
                  (255, 0, 255), 3)  ## 관심영역 그려봤음~~~~~~~~~~~~~~~~~~
    return im0,warning#warning은 경고문구 반환