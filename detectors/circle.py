import cv2
import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import least_squares


def detect_color_circles(image_path: str,
                         radius_range: Tuple[int, int] = (18, 26),
                         min_arc_length: int = 30) -> List[Dict]:
    """
    检测图片中的彩色圆环（支持遮挡）

    参数:
        image_path: 图片路径
        radius_range: 圆环半径范围 (min_radius, max_radius)
        min_arc_length: 最小圆弧长度，降低此值可检测更小的圆弧

    返回:
        检测结果列表，每个元素为字典:
        [
            {'color': 'red', 'x': 100, 'y': 150, 'radius': 20},
            {'color': 'green', 'x': 200, 'y': 250, 'radius': 22},
            ...
        ]
    """
    min_radius, max_radius = radius_range

    # HSV颜色范围定义
    color_ranges = {
        #'red': [
            #(np.array([0, 100, 100]), np.array([10, 255, 255])),
            #(np.array([170, 100, 100]), np.array([180, 255, 255]))
        #],
        'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
        'blue': [(np.array([70, 70, 100]), np.array([130, 255, 255]))]
    }

    # 读取并预处理图片
    img = image_path
    # 裁剪图片（可选）
    if img.shape[1] >= 354 and img.shape[0] >= 354:
        img = img[0:354, 0:354]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    results = []

    # 对每种颜色进行检测
    for color, ranges in color_ranges.items():
        circles = _detect_circles_by_color(hsv, ranges, min_radius, max_radius, min_arc_length)
        for x, y, r in circles:
            results.append({
                'color': color,
                'x': int(x),
                'y': int(y),
                'radius': int(r)
            })

    return results


def _detect_circles_by_color(hsv: np.ndarray,
                             color_ranges: List[Tuple],
                             min_radius: int,
                             max_radius: int,
                             min_arc_length: int) -> List[Tuple]:
    """检测特定颜色的圆（内部函数）"""
    # 创建颜色掩码
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges:
        mask |= cv2.inRange(hsv, lower, upper)

    # 形态学处理去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 混合检测：轮廓拟合 + 霍夫变换
    circles = []
    circles.extend(_detect_by_contour(mask, min_radius, max_radius, min_arc_length))
    circles.extend(_detect_by_hough(mask, min_radius, max_radius))

    # 合并相近的圆
    return _merge_circles(circles)


def _detect_by_contour(mask: np.ndarray,
                       min_radius: int,
                       max_radius: int,
                       min_arc_length: int) -> List[Tuple]:
    """基于轮廓拟合检测圆（处理遮挡情况）"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    circles = []

    for contour in contours:
        # 过滤太短的轮廓
        if cv2.arcLength(contour, False) < min_arc_length:
            continue

        points = contour.reshape(-1, 2)

        # 最小二乘拟合圆
        circle = _fit_circle_least_squares(points)
        if circle is None:
            continue

        xc, yc, r = circle

        # 验证圆的有效性
        if not (min_radius <= r <= max_radius):
            continue
        if not (0 <= xc < mask.shape[1] and 0 <= yc < mask.shape[0]):
            continue

        # 检查拟合质量
        distances = np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)
        if np.std(distances - r) < 3.0:
            circles.append((xc, yc, r))

    return circles


def _detect_by_hough(mask: np.ndarray, min_radius: int, max_radius: int) -> List[Tuple]:
    """基于霍夫圆变换检测"""
    edges = cv2.Canny(mask, 50, 150)
    circles_raw = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1,
        minDist=min_radius * 0.5, param1=50, param2=15,
        minRadius=min_radius, maxRadius=max_radius
    )

    if circles_raw is not None:
        return [(float(x), float(y), float(r)) for x, y, r in circles_raw[0]]
    return []


def _fit_circle_least_squares(points: np.ndarray) -> Tuple:
    """使用最小二乘法拟合圆"""
    if len(points) < 3:
        return None

    # 初始估计：使用质心和平均半径
    xc, yc = np.mean(points, axis=0)
    r_init = np.mean(np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2))

    # 定义残差函数
    def residuals(params):
        xc, yc, r = params
        distances = np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)
        return distances - r

    # 最小二乘优化
    result = least_squares(residuals, [xc, yc, r_init], method='lm')

    if result.success:
        xc, yc, r = result.x
        return (xc, yc, abs(r))

    return None


def _merge_circles(circles: List[Tuple], threshold: float = 15) -> List[Tuple]:
    """合并距离很近的圆（去重）"""
    if not circles:
        return []

    merged = []
    used = set()

    for i, (x1, y1, r1) in enumerate(circles):
        if i in used:
            continue

        # 找到所有相近的圆
        group = [(x1, y1, r1)]
        for j in range(i + 1, len(circles)):
            if j in used:
                continue
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < threshold:
                group.append((x2, y2, r2))
                used.add(j)

        # 取平均值
        avg_circle = tuple(np.mean(group, axis=0))
        merged.append(avg_circle)

    return merged


# ============ 使用示例 ============
if __name__ == "__main__":
    # 直接调用函数，无需实例化
    results = detect_color_circles(
        image_path="//log/test115.jpg",
        radius_range=(18, 26),
        min_arc_length=30
    )

    # # 打印结果
    # print(f"检测到 {len(results)} 个圆环:")
    # for item in results:
    #     print(f"  {item['color']:5s} 圆环: 坐标=({item['x']:3d}, {item['y']:3d}), 半径={item['radius']}")
    # print()
    # # 可选: 可视化结果
    # if results:
    #     img = cv2.imread("./log/test94.jpg")
    #     if img.shape[1] >= 354 and img.shape[0] >= 354:
    #         img = img[0:354, 0:354]
    #
    #     color_map = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}
    #
    #     for item in results:
    #         color_bgr = color_map[item['color']]
    #         cv2.circle(img, (item['x'], item['y']), item['radius'], color_bgr, 2)
    #         cv2.circle(img, (item['x'], item['y']), 3, color_bgr, -1)
    #
    #     cv2.imwrite("result.jpg", img)
    #     print("\n结果已保存到 result.jpg")