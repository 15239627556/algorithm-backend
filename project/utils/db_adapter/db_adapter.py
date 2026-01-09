import cv2
import sqlite3


def cut_roi(img, xmin, ymin, xmax, ymax, expand_scale=0.5, constant_expand=0):
    xmin_old, ymin_old, xmax_old, ymax_old = xmin, ymin, xmax, ymax
    w, h = xmax - xmin, ymax - ymin
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    size = max(w, h)
    xmin = cx - size // 2
    ymin = cy - size // 2
    xmax = cx + size // 2
    ymax = cy + size // 2
    expand = int(size * expand_scale)
    xmin = xmin - expand - constant_expand
    ymin = ymin - expand - constant_expand
    xmax = xmax + expand + constant_expand
    ymax = ymax + expand + constant_expand
    leftpad = max(0, -xmin)
    rightpad = max(0, xmax - img.shape[1])
    toppad = max(0, -ymin)
    bottompad = max(0, ymax - img.shape[0])
    imgpad = cv2.copyMakeBorder(img, toppad, bottompad, leftpad, rightpad,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if leftpad > 0:
        xmin += leftpad
        xmax += leftpad
        xmin_old += leftpad
        xmax_old += leftpad
    if toppad > 0:
        ymin += toppad
        ymax += toppad
        ymin_old += toppad
        ymax_old += toppad
    roi = imgpad[ymin:ymax, xmin:xmax].copy()
    xmin_old = xmin_old - xmin
    ymin_old = ymin_old - ymin
    xmax_old = xmax_old - xmin
    ymax_old = ymax_old - ymin
    return roi, (xmin_old, ymin_old, xmax_old, ymax_old)

# 连接到数据库
def connect_db(db_file):
    # 连接到数据库
    db_conn = sqlite3.connect(db_file)
    db_cursor = db_conn.cursor()
    return db_conn, db_cursor


# 关闭数据库
def close_db(db_conn):
    # 关闭连接
    db_conn.close()


def check_db_versions(db_cursor):
    # 执行查询，获取所有表名
    db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = db_cursor.fetchall()

    #  检查采样图信息表
    # 查找包含 'project_local_pic_info' 字段的表名
    search_term = 'project_local_pic_info'
    pic_matching_tables = [table[0] for table in tables if search_term.lower() in table[0].lower()]
    if len(pic_matching_tables) > 1:  # 正常情况不会有，若出现可检查是否有新增表
        # # print(f"存在多张包含{search_term}的表：{pic_matching_tables}，请请通过软件代码确认保存采样图信息的具体表名")
        return None
    if len(pic_matching_tables) <= 0:
        # # print(f"不存在包含{search_term}的表名，请通过软件代码确认保存采样图信息的表名")
        return None

    # 检查细胞信息表
    search_term_2 = 'project_local_doc_cell_info'
    cell_matching_tables = [table[0] for table in tables if search_term_2.lower() in table[0].lower()]
    # if len(cell_matching_tables) > 1:  # 例：从报告端下载db,用单机版审核端打开db,那么会存在两张包含'project_local_doc_cell_info'的表
    # # print(f"存在多张包含{search_term_2}的表：{cell_matching_tables}")

    if len(cell_matching_tables) <= 0:
        # # print(f"不存在包含{search_term_2}的表名，请通过软件代码确认保存医生细胞信息的表名")
        return None

    #  通过表名判断导出db数据时需要调用的子类
    if "project_local_pic_info" in pic_matching_tables and "project_local_doc_cell_info" in cell_matching_tables:  # 三代项目
        return 1
    else:  # 其它
        return 0

