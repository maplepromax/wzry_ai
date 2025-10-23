import threading
import copy
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
from ultralytics.models import YOLO
# 导入各个检测函数
from detectors.detect_teammate import detect_teammate_alive
from detectors.detect_teammate import detect_self_alive
from detectors.detect_teammate import detect_self_hp
from detectors.detect_teammate import detect_teammate_hp
from detectors.detect_map_target import detect_map_target
from detectors.detect_hp_bar import detect_hp_bars
from detectors.detect_portrait import gettx
from detectors.circle import detect_color_circles
import log


class GameState:
    """全局状态对象"""

    def __init__(self):
        self.teammate_alive = False  #队友存活信息
        self.all_alive = True  #是否全部存活
        self.teammate_position = None  #队友地图位置
        self.self_position = (0, 340)  #自己地图位置
        self.hp_bar = []  #所有人血条
        self.last_move_direction = None
        self.self_hp_bar = (831, 403)
        self.self_alive = 1  #定义为是否刚复活，1是刚复活
        self.self_hp = 0  #是否点击治疗
        self.teammate_hp = 0  #是否有队友残血

    def __repr__(self):
        return (
            f"GameState("
            f"teammate_alive={self.teammate_alive}, "
            f"all_alive={self.all_alive}, "
            f"teammate_position={self.teammate_position}, "
            f"self_position={self.self_position}, "
            f"hp_bar_len={len(self.hp_bar)}, "
            f"last_move_direction={self.last_move_direction}, "
            f"self_hp_bar={self.self_hp_bar}, "
            f"self_alive={self.self_alive})"
        )


class Detector:
    def __init__(self, frame, self_image_path, num, model_path=None, use_cuda=True):
        """
        初始化时加载模型到显卡（如果有模型）
        """
        self.last_alive = 0
        self.model = None
        self.device = 'cuda:0' if use_cuda else 'cpu'  # ✅ 保存设备信息
        if model_path:
            # ✅ 正确的 GPU 加载方式
            self.model = YOLO(model_path)

            # 方法1：使用 to() 方法（推荐）
            self.model.to(self.device)

            # 或者方法2：直接在加载时指定
            # self.model = YOLO(model_path, task='detect').to(self.device)

            print(f"✅ Model loaded on device: {next(self.model.model.parameters()).device}")
        self.self_image = cv2.imread(self_image_path)
        self.num = num
        # 状态共享
        self.game_state = GameState()
        self.lock = threading.Lock()
        self.crop_images = gettx(frame)  #队友头像

    def detect_teammate(self, frame):
        return detect_teammate_alive(frame)

    def detect_tx(self, frame):
        return gettx(frame)

    def detect_map_target(self, self_image, crop_images, frame, teammate_alive, num):
        return detect_map_target(self_image, crop_images, frame, teammate_alive, num)

    def detect_hp_bars(self, frame):
        return detect_hp_bars(self.model, frame, self.device)

    def detect_self_alive(self, frame):
        return detect_self_alive(frame)

    def detect_self_hp(self, frame):
        return detect_self_hp(frame)

    def detect_teammate_hp(self, frame):
        return detect_teammate_hp(frame)

    def detect_color_circles(self, frame):
        return detect_color_circles(frame, )

    def detect_all(self, frame):
        """
        并行执行所有检测函数，统一更新 game_state
        """
        #start = time.time()

        # 使用线程池并行执行独立检测任务
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                'self_alive': executor.submit(detect_self_alive, frame),
                'self_hp': executor.submit(detect_self_hp, frame),
                'teammate_hp': executor.submit(detect_teammate_hp, frame),
                'teammate': executor.submit(self.detect_teammate, frame),
                'hp_bars': executor.submit(self.detect_hp_bars, frame),
            }

            # 先获取 teammate_alive（地图检测需要这个参数）
            teammate_alive, all_alive = futures['teammate'].result()

            # 提交地图检测（使用已缓存的 self.crop_images）
            futures['map_target'] = executor.submit(
                self.detect_map_target,
                self.self_image,
                self.crop_images,
                frame,
                teammate_alive,
                self.num
            )

            # 获取所有结果
            self.game_state.self_alive = futures['self_alive'].result()
            self.game_state.self_hp = futures['self_hp'].result()
            self.game_state.teammate_hp = futures['teammate_hp'].result()
            hp_bars = futures['hp_bars'].result()
            map_target1, map_target2, now_map_target = futures['map_target'].result()

        # # 处理复活逻辑
        # if self.last_alive != now_alive:
        #     self.game_state.self_alive = 1
        # else:
        #     self.game_state.self_alive = 0
        # self.last_alive = now_alive

        log.Log.now_map_target = now_map_target
        log.Log.map_target = (map_target1, map_target2)
        log.Log.hp_bar = hp_bars

        has_ally_hp = any(item.get("class") == "ally_hp" for item in hp_bars)

        # 三种情况合并
        if map_target1 is not None or has_ally_hp:
            self.game_state.self_position = map_target1

        if map_target2 is not None or has_ally_hp:
            self.game_state.teammate_position = map_target2

        self.game_state.teammate_alive = teammate_alive
        self.game_state.hp_bar = hp_bars

        #print("Frame cost:", time.time() - start)
        return self.game_state

    def get_state_copy(self):
        """线程安全获取状态副本"""
        with self.lock:
            return copy.deepcopy(self.game_state)


def find_first_green(results):
    for item in results:
        if 'green' in item['color']:
            return item['x'], item['y']
    return None


def find_first_blue(results):
    for item in results:
        if 'blue' in item['color']:
            return item['x'], item['y']
    return None
