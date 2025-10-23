from heapq import heappush, heappop
from scipy.ndimage import distance_transform_edt
import math
import subprocess
import socket
import struct
import cv2
import numpy as np
import time
import threading


def find_image_in_image(template_path, target, threshold=0.8):
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

    if template is None:
        raise ValueError(f"无法读取模板图像: {template_path}")
    #if target is None:
    #raise ValueError(f"无法读取目标图像: {target_path}")

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
    if matches:
        # 按置信度降序排序，取第一个（最高的）
        max_confidence_match = max(matches, key=lambda x: x['confidence'])
        #反着返回
        result1 = (max_confidence_match['y'] + 13, max_confidence_match['x'] + 13)
        return result1
    else:
        # 没有找到匹配结果时返回None或其他标识
        return None, None


class PathPlanner:
    def __init__(self, map_image, wall_rgb, dilation_pixels=2):
        """
        初始化路径规划器
        :param map_image: 输入的地图图像 (BGR格式)
        :param wall_rgb: 墙体的RGB值，格式为 (R, G, B)
        :param dilation_pixels: 墙体膨胀的像素数量
        """
        self.map = map_image.copy()
        self.height, self.width = self.map.shape[:2]

        # 转换墙体RGB到BGR (OpenCV使用BGR格式)
        wall_bgr = (wall_rgb[2], wall_rgb[1], wall_rgb[0])

        # 创建墙体掩码 (True表示墙体)
        self.wall_mask = np.all(self.map == wall_bgr, axis=2)

        # 对墙体进行膨胀处理
        if dilation_pixels > 0:
            # 创建膨胀核
            kernel = np.ones((dilation_pixels * 2 + 1, dilation_pixels * 2 + 1), np.uint8)
            # 执行膨胀操作
            self.wall_mask = cv2.dilate(self.wall_mask.astype(np.uint8), kernel, iterations=1) > 0

        # 创建障碍物地图 (True表示可通行，False表示墙体)
        self.obstacle_map = ~self.wall_mask

        # 预计算距离变换，用于快速找到最近的可通行点
        self.distance_transform = distance_transform_edt(self.obstacle_map)

    def find_nearest_valid_point(self, pos):
        """
        找到离指定位置最近的可通行点
        :param pos: 原始位置 (x, y)
        :return: 最近的可通行点 (x, y)
        """
        x, y = pos
        # 如果已经是可通行点，直接返回
        if 0 <= x < self.height and 0 <= y < self.width and self.obstacle_map[x, y]:
            return (x, y)

        # 计算周围区域的距离变换，找到最近的可通行点
        max_dist = max(self.height, self.width)
        for dist in range(1, max_dist + 1):
            # 检查距离为dist的正方形区域
            for dx in range(-dist, dist + 1):
                for dy in [-dist, dist]:  # 上下边缘
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.height and 0 <= ny < self.width:
                        if self.obstacle_map[nx, ny]:
                            return (nx, ny)
                for dy in range(-dist + 1, dist):  # 左右边缘（排除角落，已在上面对应边缘处理）
                    for dx in [-dist, dist]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.height and 0 <= ny < self.width:
                            if self.obstacle_map[nx, ny]:
                                return (nx, ny)
        # 如果所有点都是墙体，返回None（理论上不会发生）
        return None

    def heuristic(self, a, b):
        """计算启发式距离 (欧几里得距离)"""
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_neighbors(self, pos):
        """获取8个方向的邻居节点"""
        x, y = pos
        neighbors = []
        # 8个方向: 上下左右 + 对角线
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # 检查边界和障碍物
            if 0 <= nx < self.height and 0 <= ny < self.width:
                if self.obstacle_map[nx, ny]:
                    # 对角线移动的代价是sqrt(2)，直线移动代价是1
                    cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    neighbors.append(((nx, ny), cost))
        return neighbors

    def a_star(self, start, goal):
        """
        A*路径规划算法
        :param start: 起点坐标 (x, y)
        :param goal: 终点坐标 (x, y)
        :return: 路径点列表和修正后的起点终点，如果没有路径则返回None
        """
        # 修正起点和终点位置
        corrected_start = self.find_nearest_valid_point(start)
        corrected_goal = self.find_nearest_valid_point(goal)

        # 检查修正后的点是否有效
        if corrected_start is None:
            #print("无法找到有效的起点位置！")
            return None, start, goal
        if corrected_goal is None:
            #print("无法找到有效的终点位置！")
            return None, start, goal

        # 如果起点或终点被修正，输出提示
        # if corrected_start != start:
        #     print(f"起点 {start} 位于墙体上，已修正为 {corrected_start}")
        #
        # if corrected_goal != goal:
        #     print(f"终点 {goal} 位于墙体上，已修正为 {corrected_goal}")

        # 优先队列
        open_set = []
        heappush(open_set, (0, corrected_start))

        # 记录路径
        came_from = {}

        # 记录代价
        g_score = {corrected_start: 0}
        f_score = {corrected_start: self.heuristic(corrected_start, corrected_goal)}

        while open_set:
            current = heappop(open_set)[1]

            if current == corrected_goal:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(corrected_start)
                path.reverse()
                return path, corrected_start, corrected_goal

            for neighbor, cost in self.get_neighbors(current):
                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, corrected_goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        print("未找到有效路径！")
        return None, corrected_start, corrected_goal


def plan_path(map_path, wall_rgb, start, goal, dilation_pixels=2):
    """
    主函数：执行路径规划
    :param map_path: 地图图像路径
    :param wall_rgb: 墙体的RGB值 (R, G, B)
    :param start: 起点坐标 (x, y)
    :param goal: 终点坐标 (x, y)
    :param output_path: 输出图像路径
    :param dilation_pixels: 墙体膨胀的像素数量
    :return: 绘制了路径的地图图像
    """
    # 读取地图
    map_image = cv2.imread(map_path)
    if map_image is None:
        print(f"无法读取地图文件: {map_path}")
        return None

    #print(f"地图尺寸: {map_image.shape[0]} x {map_image.shape[1]}")
    #print(f"墙体RGB: {wall_rgb}")
    #print(f"墙体膨胀像素: {dilation_pixels}")
    #print(f"原始起点: {start}, 原始终点: {goal}")

    # 创建路径规划器
    planner = PathPlanner(map_image, wall_rgb, dilation_pixels)

    # 执行A*算法
    #print("正在规划路径...")
    path, corrected_start, corrected_goal = planner.a_star(start, goal)
    if len(path) > 4:
        x = path[4][1] - path[0][1]
        y = -path[4][0] + path[0][0]
        x_ = x / math.sqrt(x * x + y * y) * 200 + 288
        y_ = -y / math.sqrt(x * x + y * y) * 200 + 860
    else:
        x_ = 288
        y_ = 860

    return x_, y_



# 使用示例
if __name__ == "__main__":
    # 参数设置
    map_path = "./map.png"  # 地图文件路径
    wall_rgb = (94, 201, 114)  # 墙体的RGB值
    start = (165, 92)  # 起点坐标 (行, 列)
    goal = (247, 231)  # 终点坐标 (行, 列)
    dilation_pixels = 3  # 墙体膨胀3个像素

    # 执行路径规划
    result = plan_path(map_path, wall_rgb, start, goal, 'test.png', dilation_pixels)

    # 可选：显示结果
    if result is not None:
        cv2.imshow('Path Planning Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
