import os
import cv2

def crop_images_in_folder(input_folder, output_folder):
    """
    批量截取文件夹内的图片指定方形区域，并保存到新文件夹。

    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        x, y: 方形区域左上角坐标
        size: 方形区域边长（单位：像素）
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    count=0
    # 支持的图片格式
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    for filename in os.listdir(input_folder):
        count=count+1
        name, ext = os.path.splitext(filename)
        if ext.lower() not in valid_ext:
            continue

        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path)

        if img is None:
            print(f"❌ 无法读取图片: {filename}")
            continue

        h, w = img.shape[:2]
        # 防止越界
        x=107
        y=207
        w=32
        h=18

        cropped = img[y:y+h, x:x+w]
        output_filename = f"{count}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, cropped)
        print(f"✅ 已裁剪并保存: {output_path}")

    print("🎉 所有图片处理完成！")

# ===== 示例调用 =====
if __name__ == "__main__":
    input_folder = "./Screenshots"   # 输入文件夹
    output_folder = "./picture"  # 输出文件夹

    # 自定义裁剪区域（左上角坐标 x, y）和边长 size
    crop_x, crop_y, crop_size = 107, 208, 300

    crop_images_in_folder(input_folder, output_folder)
