import cv2
import numpy as np
import os
from pathlib import Path


def find_image_in_image(template, target, threshold=0.8):
    """
    在目标图像中查找模板图像的位置

    参数:
        template: 模板图像 (cv2读取的图像)
        target: 目标图像 (cv2读取的图像)
        threshold: 匹配阈值 (0-1之间，越高要求越严格)

    返回:
        匹配位置列表，每个位置包含 (x, y, width, height, confidence)
    """
    if template is None or target is None:
        return []

    # 获取模板图像的尺寸
    h, w = template.shape[:2]
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

    # 去除重叠的匹配
    matches = remove_overlapping_matches(matches)

    return matches


def remove_overlapping_matches(matches, overlap_threshold=0.5):
    """去除重叠的匹配结果"""
    if len(matches) == 0:
        return []

    # 按置信度排序
    matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)

    filtered = []
    for match in matches:
        overlaps = False
        for selected in filtered:
            if is_overlapping(match, selected, overlap_threshold):
                overlaps = True
                break

        if not overlaps:
            filtered.append(match)

    return filtered


def is_overlapping(match1, match2, threshold=0.5):
    """检查两个匹配是否重叠"""
    x1, y1, w1, h1 = match1['x'], match1['y'], match1['width'], match1['height']
    x2, y2, w2, h2 = match2['x'], match2['y'], match2['width'], match2['height']

    # 计算交集
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2

    # 计算重叠率
    overlap_ratio = intersection / min(area1, area2)

    return overlap_ratio > threshold


def find_best_match_in_folder(crop_image, picture_folder, threshold=0.6):
    """
    在picture文件夹中查找与crop_image最匹配的图片

    参数:
        crop_image: 裁剪的图像 (cv2读取的图像)
        picture_folder: 包含模板图片的文件夹路径
        threshold: 匹配阈值

    返回:
        best_match: 字典 {'path': 图片路径, 'confidence': 置信度, 'image': 图像}
                   如果没找到返回 None
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    picture_path = Path(picture_folder)

    if not picture_path.exists():
        print(f"警告: 文件夹 {picture_folder} 不存在")
        return None

    image_files = [f for f in picture_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    best_match = None
    best_confidence = 0

    for template_file in image_files:
        template_path = str(template_file)
        template = cv2.imread(template_path)

        if template is None:
            continue

        try:
            matches = find_image_in_image(template, crop_image, threshold)

            if matches:
                # 获取最高置信度
                max_confidence = max(m['confidence'] for m in matches)

                if max_confidence > best_confidence:
                    best_confidence = max_confidence
                    best_match = {
                        'path': template_path,
                        'filename': template_file.name,
                        'confidence': max_confidence,
                        'image': template
                    }
        except Exception as e:
            continue

    return best_match
def gettx(image):
    """
    截取图像中指定的区域。

    参数:
        image: cv2.imread 读取的图像
        regions: 区域列表，每个区域是一个元组 (x, y, w, h)
                 x, y: 左上角坐标
                 w, h: 宽和高

    返回:
        crops: 截取后的图像列表
    """
    regions = [
        (357, 25, 408, 55),#(363, 25, 403, 51),
        (425, 25, 476, 55),#(431, 25, 471, 51)
        (493, 25, 543, 55),#(499, 25, 539, 51),
        (560, 25, 611, 55),#(567, 25, 607, 51)
    ]
    crops = []
    i=0
    for (x1, y1, x2, y2) in regions:
        i=i+1
        crop = image[ y1:y2,x1:x2]
        crop = cv2.resize(crop, (0, 0), fx=0.75, fy=0.75)
        cv2.imwrite(f"crop_{i}.jpg", crop)
        BASE_DIR = os.path.dirname(__file__)
        picture_folder=os.path.join(BASE_DIR, "picture")
        best_match = find_best_match_in_folder(crop, picture_folder, threshold=0.75)
        if best_match:
            print(f"  ✓ 找到匹配: {best_match['filename']}")
            print(f"  置信度: {best_match['confidence']:.4f}")

            # 替换为匹配的图片
            matched_image = best_match['image']
            # 调整大小与裁剪图一致
            #matched_image = cv2.resize(matched_image, (crop.shape[1], crop.shape[0]))
            crops.append(matched_image)


        else:
            print(f"  ✗ 未找到匹配，保留原图")
            crops.append(crop)

    return crops

# 示例用法
if __name__ == "__main__":
    img = cv2.imread("//log/test1.jpg")
    # 定义四个区域
    cropped_images = gettx(img)

    # 显示或保存结果
    for i, crop in enumerate(cropped_images):
        cv2.imshow(f"crop_{i}", crop)
        cv2.imwrite(f"crop_{i}.jpg", crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
