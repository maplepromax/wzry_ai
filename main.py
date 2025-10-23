import datetime
import subprocess
import socket
import struct
import sys
import traceback
import cv2
import numpy as np
import time
import threading
import os
import log
from minitouch import init_minitouch, close_minitouch
from detectors.detector import Detector
from decision import Decision
from action import ActionExecutor
from executor import Dominicap
from save_log import collect_current_params, save_to_single_json

# 全局配置
BASE_DIR = os.path.dirname(__file__)  # adb_utils.py 所在目录，也就是 myai/
ADB_PATH = os.path.join(BASE_DIR, "platform-tools", "adb.exe")
DEVICE_ID = "127.0.0.1:5555"
MINICAP_PORT = 1717  # minicap 本地端口

# 全局变量
_minicap_process = None
_minicap_socket = None
_screen_info = None

# 多线程相关
_latest_frame = None  # 最新的截图帧
_frame_lock = threading.Lock()  # 保护截图数据的锁
_capture_thread = None  # 截图线程
_stop_capture = False  # 停止标志


def run_adb_cmd(cmd_args):
    """执行 ADB 命令

    参数:
        cmd_args: 命令参数列表,例如 ["kill-server"] 或 ["connect", "emulator-5554"]
    """
    full_cmd = [ADB_PATH] + cmd_args
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)


# 程序开始前执行
adb_cmds = [
    ["kill-server"],
    ["start-server"],
    ["devices"],
    ["connect", "127.0.0.1:5555"]  # 改成你的实际设备ID
]

def init_device():
    """初始化设备连接"""
    global DEVICE_ID

    result = subprocess.run(
        [ADB_PATH, "devices"],
        capture_output=True,
        text=True
    )
    devices = result.stdout.strip().split("\n")[1:]
    online_devices = [
        line.split("\t")[0]
        for line in devices
        if line.endswith("device")
    ]

    if not online_devices:
        raise Exception("未发现已连接的设备")

    DEVICE_ID = online_devices[0]
    print(f"已选择设备：{DEVICE_ID}")


def start_minicap():
    """启动 minicap 服务"""
    global _minicap_process, DEVICE_ID

    # 获取屏幕分辨率
    result = subprocess.run(
        [ADB_PATH, "-s", DEVICE_ID, "shell", "wm", "size"],
        capture_output=True,
        text=True
    )

    # 解析分辨率，格式: "Physical size: 1080x1920"
    size_line = result.stdout.strip()
    if ":" in size_line:
        size_str = size_line.split(":")[-1].strip()
        width, height = map(int, size_str.split("x"))
    else:
        width, height = 1080, 1920  # 默认值

    print(f"屏幕分辨率: {width}x{height}")

    # 端口转发
    subprocess.run(
        [ADB_PATH, "-s", DEVICE_ID, "forward", f"tcp:{MINICAP_PORT}", "localabstract:minicap"],
        check=True
    )
    print(f"端口转发成功: {MINICAP_PORT} -> minicap")

    # 启动 minicap (确保设备上已安装 minicap)
    # 参数格式: -P 实际分辨率@输出分辨率/旋转角度
    minicap_cmd = f"LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/minicap -P {width}x{height}@{width}x{height}/0"

    _minicap_process = subprocess.Popen(
        [ADB_PATH, "-s", DEVICE_ID, "shell", minicap_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
        # stdout=subprocess.DEVNULL,  # 关键修改
        # stderr=subprocess.DEVNULL  # 关键修改
    )

    print("minicap 进程已启动，等待连接...")
    time.sleep(2)  # 等待 minicap 启动


def connect_minicap():
    """连接到 minicap socket"""
    global _minicap_socket, _screen_info

    _minicap_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _minicap_socket.connect(("127.0.0.1", MINICAP_PORT))
    _minicap_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    print("已连接到 minicap socket")

    # 读取 banner 信息 (24 字节)
    banner = _minicap_socket.recv(24)
    if len(banner) < 24:
        raise Exception("无法读取 minicap banner")

    # 解析 banner
    # 格式: version(1) length(1) pid(4) real_width(4) real_height(4)
    #       virtual_width(4) virtual_height(4) orientation(1) quirk_flag(1)
    banner_data = struct.unpack("<2B5I2B", banner)
    version = banner_data[0]
    real_width = banner_data[2]
    real_height = banner_data[3]
    virtual_width = banner_data[4]
    virtual_height = banner_data[5]

    _screen_info = {
        'version': version,
        'real_width': real_width,
        'real_height': real_height,
        'virtual_width': virtual_width,
        'virtual_height': virtual_height
    }

    print(f"minicap 信息: 真实分辨率={real_width}x{real_height}, "
          f"虚拟分辨率={virtual_width}x{virtual_height}")


def _capture_loop():
    """后台线程：持续读取 minicap 帧"""
    global _latest_frame, _minicap_socket, _stop_capture

    print("截图线程已启动")
    frame_count = 0

    while not _stop_capture:
        try:
            if not _minicap_socket:
                break

            # 读取帧头 (4 字节，表示帧数据大小)
            frame_header = _minicap_socket.recv(4)
            if len(frame_header) < 4:
                print("无法读取帧头，连接可能已断开")
                break

            frame_size = struct.unpack("<I", frame_header)[0]

            # 读取完整的 JPEG 图像数据
            jpg_data = b""
            remaining = frame_size
            while remaining > 0:
                chunk = _minicap_socket.recv(min(remaining, 4096))
                if not chunk:
                    print("连接断开")
                    _stop_capture = True
                    break
                jpg_data += chunk
                remaining -= len(chunk)

            if _stop_capture:
                break

            # 将 JPEG 数据解码为 OpenCV 图像
            img_array = np.frombuffer(jpg_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                # 更新最新帧（线程安全）
                with _frame_lock:
                    _latest_frame = img
                frame_count += 1
                if frame_count % 100 == 0:
                    #print(f"后台线程已捕获 {frame_count} 帧")
                    x = 1

        except Exception as e:
            print(f"截图线程异常: {str(e)}")
            break

    print("截图线程已退出")


def start_capture_thread():
    """启动后台截图线程"""
    global _capture_thread, _stop_capture

    _stop_capture = False
    _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
    _capture_thread.start()
    time.sleep(0.5)  # 等待第一帧
    print("后台截图线程启动成功")


def get_latest_frame():
    """获取最新的截图帧（主线程调用）"""
    with _frame_lock:
        if _latest_frame is None:
            return None
        # 返回副本，避免主线程修改影响后台线程
        return _latest_frame.copy()


def stop_capture_thread():
    """停止后台截图线程"""
    global _stop_capture, _capture_thread

    _stop_capture = True
    if _capture_thread:
        _capture_thread.join(timeout=2)
        _capture_thread = None
    print("后台截图线程已停止")


def close_minicap():
    """关闭 minicap 连接和进程"""
    global _minicap_socket, _minicap_process

    # 先停止截图线程
    stop_capture_thread()

    if _minicap_socket:
        _minicap_socket.close()
        _minicap_socket = None
        print("minicap socket 已关闭")

    if _minicap_process:
        _minicap_process.terminate()
        _minicap_process.wait()
        _minicap_process = None
        print("minicap 进程已终止")

    # 移除端口转发
    subprocess.run(
        [ADB_PATH, "-s", DEVICE_ID, "forward", "--remove", f"tcp:{MINICAP_PORT}"],
        capture_output=True
    )


#up点击的抬起动作可以改

# 主循环
if __name__ == "__main__":
    close_minicap()  # 启动前先清理旧的 minicap
    self_image_path = "./me.png"
    model_path = "./model/best.pt"
    log_path = "./logs/log.json"
    num = 1
    # log_filename = f"logs/log.txt"
    # sys.stdout = open(log_filename, "w", buffering=1, encoding="utf-8")
    # sys.stderr = sys.stdout  # 错误也写进去
    # print("程序启动，日志开始记录...")
    try:
        # 初始化
        for cmd in adb_cmds:
            run_adb_cmd(cmd)

        if not init_minitouch():
            raise Exception("MiniTouch 初始化失败")

        print("ADB 初始化完成，程序开始运行...")
        init_device()
        start_minicap()
        connect_minicap()
        # 初始化主要对象
        start_capture_thread()  # 启动后台截图线程
        screenshot = get_latest_frame()
        detector = Detector(screenshot, self_image_path, num, model_path=model_path)
        decision = Decision()
        actionexecutor = ActionExecutor()
        dominicap = Dominicap()
        # === 主循环 ===
        for i in range(10000):
            #print(f"\n第{i + 1}次操作：")
            start_time = time.time()
            screenshot = get_latest_frame()
            if screenshot is None:
                print("暂无可用截图，跳过本次循环")
                time.sleep(0.1)
                continue

            game_state = detector.detect_all(screenshot)
            #log.Log.game_state=game_state
            decision_result = decision.decide(game_state)
            result = actionexecutor.execute(decision_result, game_state)
            slide = result["slide_coords"]
            click = result["click_coords"]
            log.Log.slide_click = result
            dominicap.execute(slide, click)
            end_time = time.time()
            #cv2.imwrite(f"./log/test{i}.jpg", screenshot)
            params = collect_current_params()
            #save_to_single_json(i, params,log_path)
            time.sleep(0.03)
            #print(f"总运行时间: {end_time - start_time:.4f} 秒")

    except KeyboardInterrupt:
        print("\n用户使用 Ctrl+C 中断程序。")

    except Exception as e:
        print(f"操作失败：{str(e)}")
        traceback.print_exc()

    finally:
        # ✅ 程序退出时统一清理
        print("\n正在清理资源，请稍候...")
        close_minicap()
        close_minitouch()
        print("minicap 和 minitouch 均已安全关闭 ✅")
