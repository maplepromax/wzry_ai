import math

from funcs import plan_path


class ActionExecutor:
    def __init__(self):
        # 可以用于缓存计算结果，避免重复计算
        self.cached_positions = {}

    def execute(self, decision_result, game_state):
        """
        根据决策结果执行动作
        :param decision_result: dict, 例如 {"slide": "move_on_map", "click": "click_low_hp_teammate"}
        :param game_state: dict, 感知层提供的状态信息
        :return: dict, {"slide_coords": (x, y), "click_coords": [(x1, y1), ...]}
        """
        slide_action = decision_result.get("slide")
        click_action = decision_result.get("click")

        result = {
            "slide_coords": None,  # 默认无动作
            "click_coords": []  # 默认无点击
        }

        # -------------------------------
        # 滑动动作解析
        # -------------------------------
        if slide_action == "go_out":
            result["slide_coords"] = self.go_out()

        elif slide_action == "go_home":
            result["slide_coords"] = self.get_home_coords(game_state)

        elif slide_action == "move_to_low_hp_teammate":
            result["slide_coords"] = self.get_low_hp_teammate_coords(game_state)

        elif slide_action == "move_on_map":
            result["slide_coords"] = self.get_map_target_coords(game_state)

        elif slide_action == "move_to_teammate_hp_bar":
            result["slide_coords"] = self.get_teammate_hp_bar_coords(game_state)

        elif slide_action == "do_nothing":
            result["slide_coords"] = None

        # -------------------------------
        # 点击动作解析
        # -------------------------------
        if click_action == "click_low_hp_teammate":
            result["click_coords"] = self.get_low_hp_teammate_click_coords(game_state)

        elif click_action == "click_regular_A":
            result["click_coords"] = self.get_regular_click_A_coords(game_state)
        elif click_action == "click_regular_B":
            result["click_coords"] = self.get_regular_click_B_coords(game_state)
        elif click_action == "click_regular_C":
            result["click_coords"] = self.get_regular_click_C_coords(game_state)
        # 默认没有点击动作
        return result

    # -------------------------------
    # 下面的方法负责具体坐标计算（可自行实现）
    # -------------------------------

    def get_home_coords(self, game_state):
        # TODO: 计算回家滑动目标坐标 (x, y)
        return (183, 948)
    def go_out(self,):
        return (436,732)
    def get_low_hp_teammate_coords(self, game_state):
        # TODO: 计算血量低于0.3队友滑动目标坐标 (x, y)
        hp_bar = game_state.hp_bar
        if not hp_bar:
            return None

        # 过滤出 ally_hp 类的队友
        allies = [entry for entry in hp_bar if entry.get("class") == "ally_hp"]

        if not allies:
            return None

        # 找血量最低的队友
        lowest_hp_entry = min(allies, key=lambda x: x.get("hp", 1))
        x1, y1 = lowest_hp_entry.get("direction")
        # 返回 direction 坐标
        x = x1 - 830-200
        y = -y1 + 403+100
        x_ = x / math.sqrt(x * x + y * y) * 200 + 355
        y_ = -y / math.sqrt(x * x + y * y) * 200 + 860
        return x_, y_

    def get_map_target_coords(self, game_state):
        # TODO: 根据地图和自己位置计算滑动目标坐标 (x, y)
        x_, y_ = plan_path("./map.png", (94, 201, 114), game_state.self_position,
                           game_state.teammate_position, 2)
        return x_, y_

    def get_teammate_hp_bar_coords(self, game_state):
        # TODO: 屏幕里有队友血条时的滑动坐标 (x, y)
        hp_bar = game_state.hp_bar
        if not hp_bar:
            return None

        # 过滤出 ally_hp 类的队友
        allies = [entry for entry in hp_bar if entry.get("class") == "ally_hp"]

        if not allies:
            return None

        # 找最近的队友
        nearst_teammate = min(allies, key=lambda x: x.get("distance", 1))
        x1, y1 = nearst_teammate.get("direction")
        # 返回 direction 坐标
        x = x1 - 830
        y = -y1 + 403
        x_ = x / math.sqrt(x * x + y * y) * 200 + 355
        y_ = -y / math.sqrt(x * x + y * y) * 200 + 860
        return x_, y_

    def get_low_hp_teammate_click_coords(self, game_state):
        # TODO: 血量低队友点击坐标，返回列表 [(x1, y1), (x2, y2), ...]
        #print("test111", "得到技能坐标")
        return [(1218, 977), (1699, 635), (1692, 421), (1068, 965)]#撤退,(1861,244)

    def get_regular_click_A_coords(self, game_state):
        # TODO: 常态化点击坐标，返回列表 [(x1, y1), ...]
        #print("test111", "常态点击坐标")
        return [(1730, 155), (1602, 520), (1283, 839), (1398, 639)]  #商店、312加点

    def get_regular_click_B_coords(self, game_state):
        # TODO: 常态化点击坐标，返回列表 [(x1, y1), ...]
        #print("test111", "常态点击坐标")
        return [(1384, 944), (1713, 928),(1713, 928)]  #一技能、、普攻

    def get_regular_click_C_coords(self, game_state):
        # TODO: 常态化点击坐标，返回列表 [(x1, y1), ...]
        #print("test111", "常态点击坐标")
        return [(1495, 751)]  #二技能、进攻, (1867, 150)
