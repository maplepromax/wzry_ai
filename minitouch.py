import os
import subprocess
import socket
import time
import cv2
import numpy as np
from funcs import plan_path, find_image_in_image

# 全局变量
BASE_DIR = os.path.dirname(__file__)  # adb_utils.py 所在目录，也就是 myai/
ADB_PATH = os.path.join(BASE_DIR, "platform-tools", "adb.exe")
DEVICE_ID = "127.0.0.1:5555"


# MiniTouch 控制器
class MiniTouchController:
    def __init__(self, host='127.0.0.1', port=1111):
        self.host = host
        self.port = port
        self.sock = None
        self.max_x = 0
        self.max_y = 0
        self.max_pressure = 0
        self.is_touching = False  # 记录当前是否按下状态
        self.current_x = 0
        self.current_y = 0

    def connect(self):
        """连接到 minitouch 服务"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))

            # 读取 minitouch 的初始化信息
            banner = self.sock.recv(1024).decode('utf-8')
            print(f"MiniTouch 连接成功:\n{banner}")

            # 解析屏幕最大值
            for line in banner.split('\n'):
                if line.startswith('^ '):
                    parts = line.split()
                    self.max_x = int(parts[1])
                    self.max_y = int(parts[2])
                    self.max_pressure = int(parts[3]) if len(parts) > 3 else 50
                    print(f"屏幕分辨率: {self.max_x} x {self.max_y}, 最大压力: {self.max_pressure}")
                    break

            return True
        except Exception as e:
            print(f"连接 MiniTouch 失败: {e}")
            return False

    def send_command(self, cmd):
        """发送命令到 minitouch"""
        if self.sock:
            self.sock.sendall(cmd.encode('utf-8'))

    def down(self, x, y, contact_id=0, pressure=50):
        """按下触摸点"""
        cmd = f'd {contact_id} {int(x)} {int(y)} {pressure}\nc\n'
        self.send_command(cmd)
        self.is_touching = True
        self.current_x = x
        self.current_y = y

    def down_click(self, x, y, contact_id=0, pressure=50):
        """按下触摸点"""
        cmd = f'd {contact_id} {int(x)} {int(y)} {pressure}\nc\n'
        self.send_command(cmd)

    def move(self, x, y, contact_id=0, pressure=50):
        """移动触摸点（保持按下状态）"""
        cmd = f'm {contact_id} {int(x)} {int(y)} {pressure}\nc\n'
        self.send_command(cmd)
        self.current_x = x
        self.current_y = y

    def up(self, contact_id=0):
        """抬起触摸点"""
        cmd = f'u {contact_id}\nc\n'
        self.send_command(cmd)
        self.is_touching = False

    def up_click(self, contact_id=0):
        """抬起触摸点"""
        cmd = f'u {contact_id}\nc\n'
        self.send_command(cmd)

    def close(self):
        """关闭连接"""
        if self.is_touching:
            self.up()
        if self.sock:
            self.sock.close()
            self.sock = None
            print("MiniTouch 连接已关闭")


# 全局 MiniTouch 实例
_minitouch = None


def init_minitouch():
    """初始化 MiniTouch 连接"""
    global _minitouch

    # 启动 minitouch 服务（如果尚未启动）
    try:
        # 检查 minitouch 是否已在运行
        subprocess.run(
            [ADB_PATH, "-s", DEVICE_ID, "shell", "ps | grep minitouch"],
            capture_output=True,
            timeout=2
        )
    except:
        pass

    # 启动 minitouch（异步，不等待）
    try:
        subprocess.Popen(
            [ADB_PATH, "-s", DEVICE_ID, "shell", "/data/local/tmp/minitouch"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(0.5)  # 等待服务启动
    except Exception as e:
        print(f"启动 minitouch 服务时出错: {e}")

    # 设置端口转发
    subprocess.run(
        [ADB_PATH, "-s", DEVICE_ID, "forward", "tcp:1111", "localabstract:minitouch"],
        check=True
    )
    print("端口转发已设置: tcp:1111 -> minitouch")

    # 创建连接
    _minitouch = MiniTouchController()
    if _minitouch.connect():
        print("MiniTouch 初始化成功")
        return True
    else:
        _minitouch = None
        return False


def close_minitouch():
    """关闭 MiniTouch 连接"""
    global _minitouch
    if _minitouch:
        _minitouch.close()
        _minitouch = None


def minitouch_slide_and_hold(target_x, target_y, slide_duration=0.02, hold_duration=0.3):
    """
    滑动到指定位置并保持按下状态

    参数:
    target_x: 目标 X 坐标
    target_y: 目标 Y 坐标
    slide_duration: 滑动过程的持续时间（秒）
    hold_duration: 到达后保持按下的时间（秒）
    """
    global _minitouch

    if not _minitouch:
        print("MiniTouch 未初始化")
        return

    # 如果是第一次调用（未按下状态），先按下初始位置
    if not _minitouch.is_touching:
        # 使用初始位置 (355, 860) 作为起点
        initial_x, initial_y = 288, 860
        _minitouch.down(initial_x, initial_y)
        #print(f"按下初始位置: ({initial_x}, {initial_y})")
        time.sleep(0.005)  # 短暂延迟确保按下生效

    # 记录起始位置
    start_x = _minitouch.current_x
    start_y = _minitouch.current_y

    # 如果目标位置与当前位置相同，只需保持
    if abs(target_x - start_x) < 1 and abs(target_y - start_y) < 1:
        #print(f"位置未变化，保持在 ({target_x}, {target_y})")
        #time.sleep(hold_duration)
        return

    # 平滑滑动：分多步移动
    steps = max(10, int(slide_duration * 100))  # 根据持续时间计算步数
    step_delay = slide_duration / steps

    #print(f"从 ({start_x:.0f}, {start_y:.0f}) 滑动到 ({target_x}, {target_y})")
    #starttime = time.time()
    for i in range(1, steps + 1):
        # 线性插值计算中间位置
        progress = i / steps
        current_x = start_x + (target_x - start_x) * progress
        current_y = start_y + (target_y - start_y) * progress

        _minitouch.move(current_x, current_y)
        time.sleep(step_delay)
    #endtime = time.time()
    #print(f"总运行时间: {endtime - starttime:.4f} 秒")
    # 确保到达最终位置
    _minitouch.move(target_x, target_y)
    #print(f"到达目标位置: ({target_x}, {target_y}), 保持 {hold_duration} 秒")

    # 保持按下状态
    #time.sleep(hold_duration)


def minitouch_tap(x, y, contact_id=2, pressure=100, delay=0.05):
    """
    点击坐标 (x, y)
    :param delay: 按下后等待时间（秒），模拟轻触
    """
    #global _minitouch
    _minitouch.down_click(x, y, contact_id, pressure)
    time.sleep(delay)  # 模拟手指停留
    _minitouch.up(contact_id)


def minitouch_release():
    """抬起触摸点（停止移动）"""
    global _minitouch
    if _minitouch and _minitouch.is_touching:
        _minitouch.up_click()
        print("已抬起触摸点")
