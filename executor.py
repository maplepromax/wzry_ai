import time


class Dominicap:
    def __init__(self):
        """
        简单清晰的逻辑：
        - 有点击动作时，只执行点击，暂停滑动
        - 没有点击动作时，执行滑动
        """
        from minitouch import minitouch_slide_and_hold, minitouch_tap
        self.swipe_func = minitouch_slide_and_hold
        self.click_func = minitouch_tap

    def execute(self, swipe_coord=None, click_coords=None):
        """
        优先级逻辑：点击 > 滑动

        参数:
        swipe_coord: 单个滑动坐标 (x, y)
        click_coords: 点击坐标列表 [(x1, y1), (x2, y2), ...]

        逻辑:
        - 如果有点击动作，只执行点击（忽略滑动）
        - 如果没有点击动作，执行滑动
        """
        # 优先处理点击
        if click_coords:
            # 有点击动作，执行点击
            self._click_all(click_coords)
            if swipe_coord is not None:
                self.swipe_func(*swipe_coord)
        elif swipe_coord is not None:
            # 没有点击动作，执行滑动
            self.swipe_func(*swipe_coord)
        else:
            # 什么都没有
            pass

    def _click_all(self, click_coords):
        """依次点击每个坐标"""
        for coord in click_coords:
            self.click_func(*coord)
            time.sleep(0.01)


# 使用示例
if __name__ == "__main__":
    """
    使用示例
    """
    from minitouch import init_minitouch, close_minitouch

    # 初始化
    if not init_minitouch():
        print("初始化失败")
        exit(1)

    dom = Dominicap()

    try:
        # 示例1: 只滑动
        print("示例1: 只滑动到 (100, 200)")
        dom.execute(swipe_coord=(100, 200))
        time.sleep(1)

        # 示例2: 只点击
        print("\n示例2: 点击 (300, 400)")
        dom.execute(click_coords=[(300, 400)])
        time.sleep(1)

        # 示例3: 同时传入滑动和点击 -> 只执行点击
        print("\n示例3: 传入滑动和点击 -> 优先执行点击，忽略滑动")
        dom.execute(swipe_coord=(150, 250), click_coords=[(350, 450)])
        time.sleep(1)

        # 示例4: 多个点击
        print("\n示例4: 依次点击三个位置")
        dom.execute(click_coords=[(100, 100), (200, 200), (300, 300)])
        time.sleep(1)

        # 示例5: 连续调用（模拟游戏循环）
        print("\n示例5: 模拟游戏循环")
        for i in range(5):
            # 滑动移动
            dom.execute(swipe_coord=(100 + i * 10, 200))
            time.sleep(0.1)

            # 每3帧点击一次
            if i % 3 == 0:
                dom.execute(click_coords=[(300, 400)])

    finally:
        close_minitouch()
        print("\n测试完成")