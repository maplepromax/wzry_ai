from datetime import datetime
import json
import os
import log


def collect_current_params():
    """收集当前所有配置参数"""
    params = {
        # 可以添加更多需要保存的参数
        #"game_state": log.Log.game_state,
        "map_target": log.Log.map_target,
        "hp_bar": log.Log.hp_bar,
        "action": log.Log.action,
        "now_map_target": log.Log.now_map_target,
        "slide_click": log.Log.slide_click
    }
    return params


def save_to_single_json(i, params, log_path):
    """保存数据到单个JSON文件"""

    # 如果params是GameState对象，转换为字典
    if hasattr(params, 'to_dict'):
        params = params.to_dict()

    # 添加时间戳
    params["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params["index"] = i

    # 检查文件是否存在且不为空
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except json.JSONDecodeError:
            print(f"警告: {log_path} 文件损坏，将创建新文件")
            all_data = []
    else:
        all_data = []

    # 添加新数据
    all_data.append(params)

    # 保存回文件
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

    #print(f"数据已保存到 {log_path}")