import os

import cv2
import numpy as np


def detect_teammate_alive(image):
    """
    检测图像中指定窄条区域是否含黑色。

    参数:
        image: cv2.imread 读取的图像 (BGR)
        regions: 区域列表，每个区域是 (x1, y1, x2, y2)
                 宽度通常为1 (x2-x1=1)

    返回:
        result: 列表，每个元素为0或1，1表示区域内有蓝色
    """
    if image is None:
        raise ValueError("输入图像为空，请检查路径或格式")

    result = []

    # 将图像从 BGR 转到 HSV，更容易检测颜色
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 蓝色范围 (HSV)
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([179, 255, 30])
    regions = [
        (357, 63, 359, 68),  # 左上角、右下角坐标 (宽=2)
        (424, 63, 426, 68),
        (492, 63, 494, 68),
        (559, 63, 561, 68)
    ]
    for (x1, y1, x2, y2) in regions:
        # 防止越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        crop = hsv[y1:y2, x1:x2]

        # 创建蓝色掩码
        mask = cv2.inRange(crop, lower_blue, upper_blue)

        # 如果区域内有至少一个黑色像素，则为0
        if np.any(mask > 3):
            result.append(0)
        else:
            result.append(1)
        ifalive = 1 if any(result) else 0
    return result, ifalive


def detect_teammate_hp(img):
    """
    检查四个区域中蓝色像素的分布情况
    参数:
        img: cv2图像 (BGR格式)
    返回:
        1: 如果有任一区域的蓝色像素比例 < 0.3
        0: 如果所有区域的蓝色像素比例都 >= 0.3
    """
    regions = [
        (356, 63, 411, 69),  # 左上角、右下角坐标
        (423, 63, 478, 69),
        (491, 63, 546, 69),
        (558, 63, 613, 69)
    ]

    # 蓝色范围（HSV）
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])

    for x1, y1, x2, y2 in regions:
        # 提取ROI区域
        roi = img[y1:y2, x1:x2]

        # 转为HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 生成蓝色掩膜
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 找到非零像素坐标
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            # 区域内没有蓝色像素，result = 0.4 >= 0.3
            continue

        # 找到最右侧像素（x最大）
        idx = np.argmax(xs)
        rightmost_x = xs[idx]
        result = rightmost_x / (x2 - x1)

        # 如果任一区域 result < 0.3，立即返回 1
        if result < 0.30:
            return 1

    # 所有区域都 >= 0.3，返回 0
    return 0


def detect_self_alive(frame):
    # 获取模板图像的尺寸
    BASE_DIR = os.path.dirname(__file__)  # adb_utils.py 所在目录，也就是 myai/
    target_PATH = os.path.join(BASE_DIR,  "dead.jpg")
    target = cv2.imread(target_PATH)
    template = frame[0:860, 22:1030]

    # 使用模板匹配
    result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)

    # 找到所有匹配位置
    locations = np.where(result >= 0.8)
    matches = []
    for pt in zip(*locations[::-1]):
        matches.append({
            'x': int(pt[0]),
            'y': int(pt[1]),
        })
    if matches:
        return 0
    else:
        # 没有找到匹配结果时返回None或其他标识
        return 1


def detect_self_hp(img):
    """
        返回1表示可以放技能，0表示不放
        """
    x1, y1, x2, y2 = 925, 415, 940, 418
    roi = img[y1:y2, x1:x2]
    # 血条遮挡跳过本步骤
    # 转为HSV方便识别蓝色
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 蓝色范围（可调整）
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # 生成蓝色掩膜
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 找到非零像素坐标
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 1  # 区域内没有绿色像素
    return 0


# 示例用法
if __name__ == "__main__":
    img = cv2.imread("//log/test230.jpg")
    regions = [
        (357, 63, 359, 68),  # 左上角、右下角坐标 (宽=2)
        (424, 63, 426, 68),
        (492, 63, 494, 68),
        (559, 63, 561, 68)
    ]

    status = detect_teammate_alive(img)
    result = detect_teammate_hp(img)

    print(status)  # 例如 [1, 0, 1, 1]
    print(result)
    print(detect_self_hp(img))