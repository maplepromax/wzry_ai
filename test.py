import cv2
import numpy as np
import os
from pathlib import Path


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


def visualize_matches(target, all_matches_dict, output_path='result_batch.jpg'):
    """在目标图像上标记所有匹配位置并保存"""
    result_img = target.copy()

    colors = [
        (0, 255, 0),  # 绿色
        (255, 0, 0),  # 蓝色
        (0, 0, 255),  # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 黄色
    ]

    color_idx = 0
    for template_name, matches in all_matches_dict.items():
        color = colors[color_idx % len(colors)]
        color_idx += 1

        for i, match in enumerate(matches):
            x, y, w, h = match['x'], match['y'], match['width'], match['height']
            confidence = match['confidence']

            # 绘制矩形框
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)

            # 添加标签
            label = f"{template_name}: {confidence:.2f}"
            cv2.putText(result_img, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 保存结果
    cv2.imwrite(output_path, result_img)
    print(f"\n可视化结果已保存到: {output_path}")

    return result_img


def batch_find_images(picture_folder, target_path, threshold=0.6, output_dir='output'):
    """
    批量查找图片

    参数:
        picture_folder: 包含模板图片的文件夹路径
        target_path: 目标图像路径
        threshold: 匹配阈值
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # 获取所有图片文件
    picture_path = Path(picture_folder)
    image_files = [f for f in picture_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"在文件夹 '{picture_folder}' 中找到 {len(image_files)} 张图片")
    print(f"目标图像: {target_path}")
    print(f"匹配阈值: {threshold}")
    print("=" * 60)

    # 读取目标图像（只读一次）
    target_img = cv2.imread(target_path)
    if target_img is None:
        raise ValueError(f"无法读取目标图像: {target_path}")

    # 存储所有找到的匹配
    found_images = []
    all_matches_dict = {}

    # 逐个查找
    for idx, template_file in enumerate(image_files, 1):
        template_path = str(template_file)
        template_name = template_file.name

        try:
            print(f"\n[{idx}/{len(image_files)}] 正在查找: {template_name}")
            matches, _ = find_image_in_image(template_path, target_path, threshold)

            if matches:
                print(f"  ✓ 找到 {len(matches)} 个匹配!")
                found_images.append({
                    'filename': template_name,
                    'path': template_path,
                    'matches': matches
                })
                all_matches_dict[template_name] = matches

                # 输出详细匹配信息
                for i, match in enumerate(matches):
                    print(f"    匹配 {i + 1}: 位置({match['x']}, {match['y']}), "
                          f"置信度: {match['confidence']:.4f}")
            else:
                print(f"  ✗ 未找到匹配")

        except Exception as e:
            print(f"  ✗ 处理失败: {e}")

    # 输出总结
    print("\n" + "=" * 60)
    print(f"\n查找完成! 共找到 {len(found_images)} 张图片在目标图像中:")
    print("-" * 60)

    if found_images:
        for item in found_images:
            total_matches = len(item['matches'])
            max_confidence = max(m['confidence'] for m in item['matches'])
            print(f"  • {item['filename']}")
            print(f"    匹配次数: {total_matches}, 最高置信度: {max_confidence:.4f}")

        # 生成可视化结果
        output_path = os.path.join(output_dir, 'result_all_matches.jpg')
        visualize_matches(target_img, all_matches_dict, output_path)

        # 保存文本结果
        txt_output = os.path.join(output_dir, 'found_images.txt')
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(f"目标图像: {target_path}\n")
            f.write(f"匹配阈值: {threshold}\n")
            f.write(f"共找到 {len(found_images)} 张图片\n\n")
            for item in found_images:
                f.write(f"{item['filename']}\n")
                for i, match in enumerate(item['matches']):
                    f.write(f"  匹配{i + 1}: ({match['x']}, {match['y']}), "
                            f"置信度: {match['confidence']:.4f}\n")
                f.write("\n")
        print(f"\n文本结果已保存到: {txt_output}")
    else:
        print("  未找到任何匹配的图片")

    return found_images


# 使用示例
if __name__ == "__main__":
    # 设置路径
    picture_folder = "C:/Users/23358/PycharmProjects/wzry_ai/picture"  # 包含所有模板图片的文件夹
    target_path = "//crop_3.jpg"  # 目标图像

    # 执行批量查找
    try:
        found_images = batch_find_images(
            picture_folder=picture_folder,
            target_path=target_path,
            threshold=0.6,  # 可以调整阈值，越高越严格
            output_dir='output'  # 输出目录
        )
    except Exception as e:
        print(f"程序错误: {e}")