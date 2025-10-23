from minitouch import init_minitouch, close_minitouch
from minitouch import minitouch_tap

if not init_minitouch():
    raise Exception("MiniTouch 初始化失败")
minitouch_tap(1683, 436,pressure=100, delay=0.8)
print("ok")
