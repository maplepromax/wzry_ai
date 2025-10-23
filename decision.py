import log


class Decision:
    def __init__(self):
        self.last_direction = None
        self.counter_A = 0  # 用于常态化点击计数
        self.counter_B = 0
        self.counter_C = 0
        self.counter_Hp = 0

    def decide(self, game_state):
        """
        game_state 已经包含：
        - self_alive
        - all_alive
        - teammate_position
        - self_position
        - hp_bar  # 列表，包含所有队友血量0~1
        """
        self.counter_A += 1
        self.counter_B += 1
        self.counter_C += 1

        # -------------------------------
        # 1️⃣ 感知层预处理（计算状态）
        # -------------------------------
        self_alive = game_state.self_alive
        all_alive = game_state.all_alive
        teammate_pos = game_state.teammate_position
        self_pos = game_state.self_position
        hp_bar = game_state.hp_bar
        has_en_hp = any(item.get("class") == "enemy_hp" for item in hp_bar)
        # 检查血量低于0.3的队友
        #print("testhp",hp_bar)
        low_hp_teammates = [h for h in hp_bar if h["hp"] < 0.35]
        has_low_hp = len(low_hp_teammates) > 0
        has_team_hp = any(item.get("class") == "ally_hp" for item in hp_bar)
        x = 0
        # -------------------------------
        # 2️⃣ 滑动事件条件-优先级
        # -------------------------------
        priorities = [
            (100, lambda: x == 1, lambda: "go_out"),  # 刚复活先走几步(暂时不用这个逻辑)
            (90, lambda: all_alive == 0, lambda: "go_home"),  # 队友都死了回家
            (80, lambda: has_low_hp, lambda: "move_to_low_hp_teammate"),  # 血量低于0.3队友移动
            (70, lambda: teammate_pos is not None and self_pos is not None,
             lambda: "move_on_map"),  # 双方位置都存在按地图移动
            (60, lambda: (teammate_pos is None or self_pos is None) and has_team_hp,
             lambda: "move_to_teammate_hp_bar"),  # 一方位置不存在，朝血条移动
            (50, lambda: (teammate_pos is None or self_pos is None) and not has_team_hp and not has_en_hp,
             lambda: "do_nothing"),  # 一方位置不存在回家
            (40, lambda: (teammate_pos is None or self_pos is None) and not has_team_hp and has_en_hp,
             lambda: "go_home")  # 一方位置不存在且盟友血条存在时回家

        ]

        # 执行最高优先级满足条件的滑动动作
        for score, cond_func, action_func in sorted(priorities, key=lambda x: -x[0]):
            if cond_func():
                slide_action = action_func()
                break
        else:
            slide_action = "do_nothing"  # 安全兜底
        # -------------------------------
        # 3️⃣ 点击事件
        # -------------------------------
        click_action = None
        if (has_low_hp and game_state.teammate_hp == 1) or (
                game_state.self_hp == 1 and self_alive == 1):  #队友血低且头像血低/自己血低就点
            click_action = "click_low_hp_teammate"
            # --- 三个常态点击模式 ---
        else:
            self.counter_Hp = 0
            # 每个模式可以设不同触发轮次
            if self.counter_A >= 51:  # 每10轮触发一次A
                click_action = "click_regular_A"
                self.counter_A = 0
            elif self.counter_B >= 59:  # 每15轮触发一次B
                click_action = "click_regular_B"
                self.counter_B = 0
            elif self.counter_C >= 79:  # 每20轮触发一次C
                click_action = "click_regular_C"
                self.counter_C = 0

        # -------------------------------
        # 4️⃣ 返回动作
        # -------------------------------
        #print("action", slide_action)
        log.Log.action = (slide_action, click_action)
        return {
            "slide": slide_action,
            "click": click_action
        }

    def find_close_enemy(self, enemies):
        return min(enemies, key=lambda e: e["distance"])

    def select_teammate(self, teammates):
        # 可加更多规则，比如优先最近或血量多的
        return teammates[0]

    def compute_direction(self, target):
        # 根据目标坐标计算滑动方向
        return (target["x"], target["y"])
