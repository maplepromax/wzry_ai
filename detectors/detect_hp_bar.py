import time

import cv2
import numpy as np
from ultralytics import YOLO


def find_rightmost_blue_pixel(img, rect):
    """
    在给定矩形区域内查找最右侧的蓝色像素点
    参数:
        img: cv2图像 (BGR格式)
        rect: (x1, y1, x2, y2) 左上角和右下角坐标
    返回:
        (x, y) 最右侧蓝色像素的坐标，如果未找到则返回 None
    """

    x1, y1, x2, y2 = map(int, rect)
    y2_ = int(y1 + (y2 - y1) / 2)   #抬高防止识别蓝条
    roi = img[y1:y2_, x1:x2]
    if x2 - x1 < 110:
        return 0.4      #血条遮挡跳过本步骤
    # 转为HSV方便识别蓝色
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 蓝色范围（可调整）
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])

    # 生成蓝色掩膜
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 找到非零像素坐标
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.4  # 区域内没有蓝色像素

    # 找到最右侧像素（x最大）
    idx = np.argmax(xs)
    rightmost_x = xs[idx] + x1
    result = (rightmost_x - x1 - 40) / (x2 - x1 - 40)

    return result


def detect_hp_bars(model, frame,device='cuda:0'):
    """
    使用已加载的 YOLO 模型对一帧图像进行目标检测。

    参数：
        model: 已加载的 YOLO 模型对象 (YOLO('best.pt'))
        frame: 已加载的 OpenCV 图像 (BGR 格式)

    返回：
        results_list: [
            {"class": "类别名", "confidence": 置信度, "bbox": [x1, y1, x2, y2]},
            ...
        ]
    """
    # 推理
    #print(f"Model device: {next(model.model.parameters()).device}")
    results = model.predict(
        source=frame,
        verbose=False,
        device=device,  # 明确指定设备
        half=True if 'cuda' in str(device) else False,  # GPU时使用FP16加速
    )

    x0, y0 = (831, 403)
    results_list = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])  # 类别ID
        conf = float(box.conf[0])  # 置信度
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # 坐标（映射回原图）
        class_name = results[0].names[cls_id]  # 类别名称

        # 只保留置信度大于0.4的结果
        if conf > 0.63:
            distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if class_name == "ally_hp":
                hp = find_rightmost_blue_pixel(frame, [x1, y1, x2, y2])
            else:
                hp = 1
            results_list.append({
                "class": class_name,
                "distance": distance,
                "direction": [x1, y1],
                "hp": hp,
                "box": [x1, y1, x2, y2]
                #"confidence": conf  # 可以顺便存一下置信度，方便调试
            })

    return results_list


if __name__ == "__main__":
    model_path = r"C:\Users\23358\PycharmProjects\yolo\runs\train_custom\exp_yolov8n3\weights\best.pt"  # 模型路径
    img_path = r"C:\Users\23358\PycharmProjects\myai\log\test58.jpg"  #
    img = cv2.imread(img_path)

    # 加载模型
    model = YOLO(model_path)
    model.to("cuda:0")
    starttime=time.time()
    results_list = detect_hp_bars(model, img)
    print(time.time()-starttime)
    print(results_list)
