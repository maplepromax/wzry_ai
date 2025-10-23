import cv2
import numpy as np


def find_image_in_image(template_path, target_path, threshold=0.8):
    """
    在目标图像中查找模板图像的位置

    参数:
        template_path: 模板图像路径 (图a)
        target_path: 目标图像路径 (图b)
        threshold: 匹配阈值 (0-1之间，越高要求越严格)

    返回:
        匹配位置列表，每个位置包含 (x, y, width, height, confidence)
    """
    # 读取图像
    template = cv2.imread(template_path)
    target = cv2.imread(target_path)
    if template is None:
        raise ValueError(f"无法读取模板图像: {template_path}")
    if target is None:
        raise ValueError(f"无法读取目标图像: {target_path}")

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

    # 去除重叠的匹配（非极大值抑制）
    matches = remove_overlapping_matches(matches)

    return matches, target


def remove_overlapping_matches(matches, overlap_threshold=0.5):
    """去除重叠的匹配结果"""
    if len(matches) == 0:
        return []

    # 按置信度排序
    matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)

    filtered = []
    for match in matches:
        # 检查是否与已选择的匹配重叠
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


def visualize_matches(target, matches, output_path='result.jpg'):
    """在目标图像上标记匹配位置并保存"""
    result_img = target.copy()

    for i, match in enumerate(matches):
        x, y, w, h = match['x'], match['y'], match['width'], match['height']
        confidence = match['confidence']

        # 绘制矩形框
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 添加置信度标签
        label = f"#{i + 1}: {confidence:.2f}"
        cv2.putText(result_img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 保存结果
    cv2.imwrite(output_path, result_img)
    print(f"结果已保存到: {output_path}")

    return result_img


# 使用示例
if __name__ == "__main__":
    # 设置图像路径
    #template_path = "me.png"  # 模板图像（图a）
    template_path = "//picture/54.jpg"  # 模板图像（图a）
    target_path = "//log/test0.jpg"  # 目标图像（图b）

    try:
        # 查找匹配
        matches, target_img = find_image_in_image(template_path, target_path, threshold=0.7)
        print(matches)
        # 输出结果
        if matches:
            print(f"找到 {len(matches)} 个匹配:")
            for i, match in enumerate(matches):
                print(f"  匹配 {i + 1}:")
                print(f"    位置: ({match['x']}, {match['y']})")
                print(f"    大小: {match['width']} x {match['height']}")
                print(f"    置信度: {match['confidence']:.4f}")
        else:
            print("未找到匹配")

        # 可视化并保存结果
        visualize_matches(target_img, matches, 'result.jpg')

    except Exception as e:
        print(f"错误: {e}")