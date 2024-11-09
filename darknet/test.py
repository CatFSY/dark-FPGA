import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel
import cv2
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='darknet/data',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='darknet/results', help='location of the data corpus')
parser.add_argument('--model', type=str, default='darknet/weights/difficult.pt', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()

def main():
    # 检查是否有可用的 GPU 设备
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)

    # 加载模型
    model = Finetunemodel(args.model)
    model = model.cuda()
    model.eval()

    # 打开摄像头
    cap = cv2.VideoCapture(0)  # 0 代表默认摄像头
    if not cap.isOpened():
        print("Could not open camera.")
        sys.exit(1)

    print("Press 'q' to quit.")

    with torch.no_grad():
        while True:
            # 读取摄像头的一帧
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            frame = cv2.resize(frame, (320, 240))  # 注意 OpenCV 的 resize 接受的是 (宽, 高) 参数

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_chw = np.transpose(frame_rgb, (2, 0, 1)) / 255.0
            frame_batch = np.expand_dims(frame_chw, axis=0)
            frame_tensor = torch.FloatTensor(frame_batch).cuda()  # 将输入张量移动到 GPU
            # print(frame_tensor.shape)
            i, r,fea1,fea2 = model(frame_tensor)
            processed_frame = fea1.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            # processed_frame = (processed_frame * 255).astype(np.uint8)  # 如果输出是 0-1 之间，则乘以 255
            # print(processed_frame.shape)
            # 转换为 OpenCV 显示格式（BGR）
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('yuan', frame)
            cv2.imshow('Processed Frame', processed_frame_bgr)
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
