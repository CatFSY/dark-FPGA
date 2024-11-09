import cv2
import torch
import numpy as np
from django.http import StreamingHttpResponse
from django.shortcuts import render
import queue
import threading
import json
from django.http import JsonResponse
from darknet.model import Finetunemodel
image_queue_original = queue.Queue()
image_queue_feature_one = queue.Queue()
image_queue_feature_two = queue.Queue()
image_queue_output = queue.Queue()
# 加载模型
model = Finetunemodel('darknet/weights/difficult.pt')
model = model.cuda()
model.eval()
import sys
# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 代表默认摄像头
if not cap.isOpened():
    print("Could not open camera.111111111111111111111111111111")
    sys.exit(1)

queue_status = {
    'image_queue_original': False,
    'image_queue_feature_one': False,
    'image_queue_feature_two': False,
    'image_queue_output': False
}
queue_status_control = {
    'image_queue_original': False,
    'image_queue_feature_one': False,
    'image_queue_feature_two': False,
    'image_queue_output': False
}
def control_queues(request):
    global queue_status
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            queue_status["image_queue_original"] = data['origin'] 
            queue_status["image_queue_feature_one"] = data['feature_one']
            queue_status["image_queue_feature_two"] = data['feature_two']
            queue_status["image_queue_output"] = data['output']
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        return JsonResponse({'status': 'error', 'message': 'Only POST requests are allowed'})
# 视频流生成器
def frame_generator():
    global queue_status
    while True:
        # print(1)
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        # frame = cv2.imread("darknet/data/99.png")
        # 处理视频帧
        frame_resized = cv2.resize(frame, (320, 240))  # 调整帧大小
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_chw = np.transpose(frame_rgb, (2, 0, 1)) / 255.0
        frame_batch = np.expand_dims(frame_chw, axis=0)
        frame_tensor = torch.FloatTensor(frame_batch).cuda()
        with torch.no_grad():
            ori, out,fea1,fea2 = model(frame_tensor)
            if queue_status['image_queue_original']:
                image_queue_original.put(ori)
            if queue_status['image_queue_feature_one']:
                image_queue_feature_one.put(fea1)
            if queue_status['image_queue_feature_two']:
                image_queue_feature_two.put(fea2)
            if queue_status['image_queue_output']:
                image_queue_output.put(out)
def start_frame_generator_thread(request):
    thread = threading.Thread(target=frame_generator)
    thread.daemon = True  # 使线程在主程序退出时自动退出
    thread.start()
    return JsonResponse({'status': 'success', 'message': 'Video stream started'})
def generate_original_frame():
    while True:
        # 从原视频队列获取最新的一帧图像并转换为JPEG格式
        image = image_queue_original.get() if not image_queue_original.empty() else None
        if image is not None:
            processed_frame = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)*255
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', processed_frame_bgr)
            if ret:
                frame = jpeg.tobytes()  # 转换为字节流
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def generate_feature_one_frame():
    while True:
        # 从特征一队列获取最新的一帧图像并进行特征一处理
        image = image_queue_feature_one.get() if not image_queue_feature_one.empty() else None
        if image is not None:
            # 特征一处理
            processed_frame = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)*255
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', processed_frame_bgr)
            if ret:
                frame = jpeg.tobytes()  # 转换为字节流
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def generate_feature_two_frame():
    while True:
        # 从特征二队列获取最新的一帧图像并进行特征二处理
        image = image_queue_feature_two.get() if not image_queue_feature_two.empty() else None
        if image is not None:
            # 特征二处理
            processed_frame = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)*255
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', processed_frame_bgr)
            if ret:
                frame = jpeg.tobytes()  # 转换为字节流
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def generate_output_frame():
    while True:

        image = image_queue_output.get() if not image_queue_output.empty() else None
        if image is not None:
            # 输出处理
            processed_frame = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)*255
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', processed_frame_bgr)
            if ret:
                frame = jpeg.tobytes()  # 转换为字节流
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# 视频流视图
def video_stream_output(request):
    return StreamingHttpResponse(generate_output_frame(), content_type='multipart/x-mixed-replace; boundary=frame')

def video_stream_fea2(request):
    return StreamingHttpResponse(generate_feature_two_frame(), content_type='multipart/x-mixed-replace; boundary=frame')
def video_stream_fea1(request):
    return StreamingHttpResponse(generate_feature_one_frame(), content_type='multipart/x-mixed-replace; boundary=frame')
def video_stream_ori(request):
    return StreamingHttpResponse(generate_original_frame(), content_type='multipart/x-mixed-replace; boundary=frame')