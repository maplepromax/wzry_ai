from ultralytics import YOLO
import cv2


def detect_objects(model, frame):
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
    results = model.predict(source=frame, verbose=False)

    results_list = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])  # 类别ID
        conf = float(box.conf[0])  # 置信度
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # 坐标（映射回原图）
        class_name = results[0].names[cls_id]  # 类别名称

        results_list.append({
            "class": class_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    return results_list


if __name__ == "__main__":
    model_path = r"C:\Users\23358\PycharmProjects\yolo\runs\train_custom\exp_yolov8n3\weights\best.pt"  # 模型路径
    img_path = "./log/test2138.jpg"  #
    img = cv2.imread(img_path)

    # 加载模型
    model = YOLO(model_path)
    results_list = detect_objects(model, img )
    print(results_list)
