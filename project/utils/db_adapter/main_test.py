import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from db_adapter import connect_db, close_db, check_db_versions
from db_old import ZWXKProjectDBSon1
from db_new import ZWXKProjectDBSon0


def find_db_files(root_folder):
    root_path = Path(root_folder)
    return [str(file) for file in root_path.rglob('*.db')]


def export_x100_have_image_test():
    for item in zwxkProjectDbSon.get_x100_sample_images():
        print(item.x, item.y, item.w, item.h)


def export_x100_big_image_test():
    for item in zwxkProjectDbSon.get_x100_meg_images():
        print(item.x, item.y, item.w, item.h)


def export_x40_image_test():
    for item in zwxkProjectDbSon.get_x40_sample_images():
        print(item.row, item.col, item.x, item.y, item.w, item.h)


def export_x100_image_and_x40_rect_image_test():
    # 获取x100有核采样图以及对应x40分布图区域截图
    for item in zwxkProjectDbSon.get_x100_have_and_x40Area():
        # 指定保存路径
        save_dir = r'E:\image_data\tt'  # 可以自定义多层目录
        save_dir = os.path.join(
            os.path.join(os.path.join(os.path.join(save_dir, last_folder), str(item.smearNo)), "有核"),
            str(item.md5))
        save_path_x100 = os.path.join(save_dir, f"{str(item.md5)}.png")
        save_path_x40 = os.path.join(save_dir, f"{str(item.md5)}-x40.png")
        # 如果目录不存在，则创建
        os.makedirs(save_dir, exist_ok=True)
        img_pil_x100 = Image.fromarray(item.imageX100)
        img_pil_x40 = Image.fromarray(item.imageX40)
        img_pil_x100.save(save_path_x100, format='png')  # 保存采样图
        img_pil_x40.save(save_path_x40, format='png')
        print(item.instance_list)


def export_x100_big_image_and_x40_rect_image_test():
    # 获取x100巨核采样图以及对应x40分布图区域截图
    for item in zwxkProjectDbSon.get_x100_big_and_x40Area():
        # 指定保存路径
        save_dir = r'E:\image_data\tt'  # 可以自定义多层目录
        save_dir = os.path.join(
            os.path.join(os.path.join(os.path.join(save_dir, last_folder), str(item.smearNo)), "巨核"), str(item.md5))
        save_path_x100 = os.path.join(save_dir, f"{str(item.md5)}.png")
        save_path_x40 = os.path.join(save_dir, f"{str(item.md5)}-x40.png")
        # 如果目录不存在，则创建
        os.makedirs(save_dir, exist_ok=True)
        img_pil_x100 = Image.fromarray(item.imageX100)
        img_pil_x40 = Image.fromarray(item.imageX40)
        img_pil_x100.save(save_path_x100, format='png')  # 保存采样图
        img_pil_x40.save(save_path_x40, format='png')


def export_smear_chenck_type_test():
    dict = zwxkProjectDbSon.get_smear_type()
    print(dict)


if __name__ == '__main__':
    db_files = find_db_files(r"E:\db")
    progress_bar = tqdm(range(len(db_files)), desc='Processing')
    for index, item in enumerate(db_files):
        cellid = 0
        # 获取目录部分
        dir_path = os.path.dirname(item)

        # 获取最后一层文件夹名
        last_folder = os.path.basename(dir_path)

        # 连接到数据库
        conn, cursor = connect_db(item)
        # 判断db版本
        sonClassNO = check_db_versions(cursor)
        zwxkProjectDbSon = None
        if sonClassNO == 0:
            zwxkProjectDbSon = ZWXKProjectDBSon0(cursor)
        elif sonClassNO == 1:
            zwxkProjectDbSon = ZWXKProjectDBSon1(cursor)
        else:
            print("未知版本")
        if zwxkProjectDbSon is not None:
            export_smear_chenck_type_test()

        #  关闭数据连接
        close_db(conn)
        progress_bar.update(1)

    progress_bar.close()
