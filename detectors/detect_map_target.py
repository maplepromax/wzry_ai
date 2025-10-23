import cv2
import numpy as np
from heapq import heappush, heappop
from scipy.ndimage import distance_transform_edt
import math


def crop_cv2_image(img):
    """截取cv2格式图像的矩形区域"""
    x1, y1 = 0, 0
    x2, y2 = 343, 400

    try:
        if img is None:
            raise ValueError("输入图像为空")
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            raise ValueError("坐标超出图像范围")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("无效的坐标值")

        return img[y1:y2, x1:x2]

    except Exception as e:
        print(f"图像截取失败: {str(e)}")
        return None


def find_image_in_image(template, target, threshold=0.81):
    """
    在目标图像中查找模板图像的位置

    参数:
        template: 模板图像cv2数据
        target_path: 目标图像路径 (图b)
        threshold: 匹配阈值 (0-1之间，越高要求越严格)

    返回:
        匹配位置列表，每个位置包含 (x, y, width, height, confidence)
    """
    # 获取模板图像的尺寸
    h, w = template.shape[:2]
    target = crop_cv2_image(target)
    # 使用模板匹配
    result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)

    # 找到所有匹配位置
    locations = np.where(result >= threshold)
    matches = []

    for pt in zip(*locations[::-1]):
        confidence = result[pt[1], pt[0]]
        matches.append({
            'x': int(pt[0]),
            'y': int(pt[1]),
            'width': w,
            'height': h,
            'confidence': float(confidence)
        })
    if matches:
        # 按置信度降序排序，取第一个（最高的）
        max_confidence_match = max(matches, key=lambda x: x['confidence'])
        #反着返回
        result1 = (
            min(max_confidence_match['y'] + int(h / 2), 345),
            min(max_confidence_match['x'] + int(w / 2), 345)
        )
        #print("test22",max_confidence_match)
        return result1
    else:
        # 没有找到匹配结果时返回None或其他标识
        return None


def detect_map_target(self_image, crop_images, frame, teammate_alive, num):  #先手动设置跟谁
    """
        frame: 地图cv2 图片
        teammate_alive: 队友是否存活，4个数字列表 [1,0,1,1]
        num: 1-4，暂时手动设置优先跟谁
        使用全局变量：
            self_image: 自己头像，蔡文姬
            crop_images: 4张角色图像列表
        返回: (自己位置, 另一个位置) 坐标，如果没找到返回 None
        """

    # 查找自己
    self_pos = find_image_in_image(self_image, frame)
    x = 0
    # 查找另一个存活角色
    other_pos = None
    if teammate_alive[num - 1] == 1:
        # 目标存活，用对应的crop_images查找
        #print("yyyysss")
        other_pos = find_image_in_image(crop_images[num - 1], frame)
        x = -1
    else:
        # 目标死亡，从左到右查找其他存活角色
        for i in range(4):
            if i == num - 1:
                continue
            if teammate_alive[i] == 1:
                other_pos = find_image_in_image(crop_images[i], frame)
                if other_pos is not None:
                    x = i
                    break
    #print("position11", self_pos, other_pos)
    return self_pos, other_pos, x


if __name__ == "__main__":
    # 设置图像路径
    #template_path = "C:/Users/23358/PycharmProjects/wzry_ai/me.png"  # 模板图像（图a）
    template_path = "//crop_3.jpg"  # 模板图像（图a）
    target_path = "//log/test478.jpg"  # 目标图像（图b）
    template = cv2.imread(template_path)
    target = cv2.imread(target_path)
    #print(template, target)
    result = find_image_in_image(template, target, threshold=0.3)
    print(result)
