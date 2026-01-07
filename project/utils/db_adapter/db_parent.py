import math
import numpy as np
import io
from PIL import Image
import re


# 分布图像素坐标，左上角为(0, 0)
# 分布图行列坐标，左下角为(0, 0)
# 细胞列表instance_list [x, y, w, h, m_umodify_type, m_ucell_type1]
class ImageX40(object):
    def __init__(self, image_arr, md5, smearNo, row, col, x, y, w, h):
        self.image = image_arr
        self.md5 = md5
        self.smearNo = smearNo
        self.row = row
        self.col = col
        self.x = x  # 在分布图上坐标左上x
        self.y = y  # 在分布图上坐标左上y
        self.w = w  # 图片宽
        self.h = h  # 图片高


class ImageX100(object):
    def __init__(self, image_arr, md5, smearNo, instance_list, x, y, w, h):
        self.image = image_arr
        self.md5 = md5
        self.smearNo = smearNo
        self.instance_list = instance_list
        self.x = x  # 在分布图上坐标左上x
        self.y = y  # 在分布图上坐标左上y
        self.w = w  # 图片宽
        self.h = h  # 图片高


class ImageX100AndX40(object):
    def __init__(self, image_arr_x100, image_arr_x40, md5, smearNo, instance_list):
        self.imageX100 = image_arr_x100
        self.imageX40 = image_arr_x40
        self.md5 = md5
        self.smearNo = smearNo
        self.instance_list = instance_list


class X40ImageBigCell(object):
    def __init__(self, image_arr, md5, smearNo, instance_list, photoType):
        self.image = image_arr
        self.md5 = md5
        self.smearNo = smearNo
        self.instance_list = instance_list
        self.photoType = photoType  # 该巨核细胞最终以什么形式展示在采样图 5-x40巨核截图 6-x100百倍巨核拍摄


class ZWXKProjectDB(object):
    def __init__(self, db_cursor):
        self._cursor = db_cursor

    def _blob_to_u8(self, m_blob, start):
        bytes_data = bytes.fromhex(m_blob[start:start + 1].hex())
        u8 = int.from_bytes(bytes_data, byteorder='big')
        return u8

    def _blob_to_u16(self, m_blob, start):
        bytes_data = bytes.fromhex(m_blob[start:start + 2].hex())
        u16 = int.from_bytes(bytes_data, byteorder='big')
        return u16

    def _blob_to_u32(self, m_blob, start):
        bytes_data = bytes.fromhex(m_blob[start:start + 4].hex())
        u32 = int.from_bytes(bytes_data, byteorder='big')
        return u32

    # 计算数据块总数
    def _calc_block_num(self, md5):
        self._cursor.execute('SELECT TableIndex FROM project_local_photo_data')
        row = self._cursor.fetchone()  # 获取一行数据
        tableIndex_len = len(row[0])
        # 根据md5值查询图片数据总字节数
        tableIndex = md5 + b'\x00' * (tableIndex_len - 44)
        self._cursor.execute('SELECT FileLen FROM project_local_photo_data WHERE TableIndex = ?', (tableIndex,))
        x40_level_image_data_len = self._cursor.fetchall()
        x40_level_image_data_len = self._blob_to_u32(x40_level_image_data_len[0][0], 0)
        # 计算数据块总数
        if tableIndex_len == 132:
            block_num = math.ceil(x40_level_image_data_len / 2048)
        else:
            block_num = math.ceil(x40_level_image_data_len / 40960)
        return block_num

    def _calc_tableIndex(self, block, md5, TableIndex_len):
        # 将整数转换为字节串
        byte_data = block.to_bytes(4, byteorder='big')
        if TableIndex_len == 50:
            # md5 + 数据块下标
            tableIndex = md5 + byte_data
            # 获取 tableIndex 数据的长度
            current_length = len(tableIndex)
            # 检查长度是否达到期望的长度
            if current_length < 50:
                # 计算需要追加的零字节数量
                num_zeros_to_add = 50 - current_length
                # 追加零字节
                tableIndex = tableIndex + b'\x00' * num_zeros_to_add
        else:
            # 获取 tableIndex 数据的长度
            current_length = len(md5)
            # 检查长度是否达到期望的长度
            if current_length < 128:
                # 计算需要追加的零字节数量
                num_zeros_to_add = 128 - current_length
                # 追加零字节
                md5 = md5 + b'\x00' * num_zeros_to_add
            tableIndex = md5 + byte_data
        return tableIndex

    def _query_image(self, block_num, md5):
        self._cursor.execute('SELECT TableIndex FROM project_local_photo_data')
        row = self._cursor.fetchone()  # 获取一行数据
        TableIndex_len = len(row[0])

        # 循环取图片数据
        image_data = b''
        for block in range(block_num):
            tableIndex = self._calc_tableIndex(block, md5, TableIndex_len)
            # 查询当前数据块长度 及图片数据
            self._cursor.execute('SELECT DataLen, Data FROM project_local_photo_data WHERE TableIndex = ?',
                                 (tableIndex,))
            datas = self._cursor.fetchall()[0]
            dataLen = datas[0]
            image_data_block = datas[1]
            dataLen = self._blob_to_u32(dataLen, 0)
            # 根据当前数据块长度取图片数据
            image_data += image_data_block[:dataLen]
        return image_data

    def _blob_to_numpy(self, blob_image_data):
        pil_image = Image.open(io.BytesIO(blob_image_data))
        # 将 PIL 图像转换为 RGB 格式
        pil_image_rgb = pil_image.convert('RGB')
        image_np = np.array(pil_image_rgb)
        return image_np

    # 检查字段是否存在
    def column_exists(self, table_name, column_name):
        self._cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in self._cursor.fetchall()]
        return column_name in columns

    # 获取所有玻片的所有指定层40x平扫图
    def get_x40_sample_images_(self, keep_ratio=1):
        table_name = "project_local_tif_pic_info"
        self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if self._cursor.fetchone() is None:
            # print(f"不存在表 {table_name}, 应是未下载x40平扫图")
            return None
        self._cursor.execute('SELECT Data, SmearNo, XPos, YPos FROM project_local_tif_pic_info WHERE Level = ?',
                             (keep_ratio,))
        for i_index, i in enumerate(self._cursor.fetchall()):
            md5_, smearNo, col, row = i
            # 截取md5
            md5 = md5_[:44]
            image_w = self._blob_to_u32(md5_, 128)
            image_h = self._blob_to_u32(md5_, 132)
            block_num = self._calc_block_num(md5)
            if block_num <= 0:
                # print(f'md5:{md5}没有图片数据')
                imageX40 = ImageX40(np.empty(0), md5, smearNo, 0, 0, 0, 0, 0, 0)
            else:
                # 根据行列号计算x40在分布图上的坐标
                max_ypos_dict = self.get_max_row()
                if max_ypos_dict is None:
                    x = y = 0
                else:
                    x = 2448 * col
                    max_row = max_ypos_dict.get(smearNo)
                    y = 2048 * (max_row - row)
                image_data = self._query_image(block_num, md5)
                if image_data == b'':
                    # print(f'md5:{md5}获取图片数据失败')
                    imageX40 = ImageX40(np.empty(0), md5, smearNo, 0, 0, 0, 0, 0, 0)
                else:
                    imageX40 = ImageX40(self._blob_to_numpy(image_data), md5, smearNo, row, col, x, y, image_w, image_h)

            yield imageX40

    # 根据行列号获取平扫图MD5
    def get_x40_image_by_row_col(self, smearNo, row, col):
        table_name = "project_local_tif_pic_info"
        self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if self._cursor.fetchone() is None:
            # print(f"不存在表 {table_name}, 应是未下载x40平扫图")
            return None
        self._cursor.execute('SELECT Data FROM project_local_tif_pic_info WHERE SmearNo = ? AND XPos = ? AND YPos = ?',
                             (smearNo, col, row))
        data = self._cursor.fetchall()[0]
        md5 = data[0]
        return md5

    # 获取x40巨核细胞所在平扫图
    def get_x40_big_cell_image(self):
        self._cursor.execute(
            'SELECT SmearNo, MAX(YPos) '
            'FROM project_local_tif_pic_info '
            'GROUP BY SmearNo'
        )
        rows = self._cursor.fetchall()
        max_ypos_dict = {s_no: max_y for s_no, max_y in rows}

        big_cell_image_list = self.get_big_cell_image_infos()

        for index, i in enumerate(big_cell_image_list):
            md5_, smearNo, m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h, image_w, image_h, photoType = i

            # 计算该采样图所在x40平扫图的行列号
            row, rem_y = divmod(m_upixel_y, 2048)
            col, rem_x = divmod(m_upixel_x, 2448)

            row = max_ypos_dict.get(smearNo) - row
            # 获取该巨核所在x40平扫图MD5
            md5 = self.get_x40_image_by_row_col(smearNo, row, col)
            # 计算巨核细胞框转回到分布图的比例
            if photoType == 5:
                w_ratio = 1
                h_ratio = 1
            else:
                w_ratio = float(m_upixel_w) / float(image_w)
                h_ratio = float(m_upixel_h) / float(image_h)
            # 获取采样图上的细胞信息
            big_cell_infos = self._query_big_cell_infos(md5_, w_ratio, h_ratio, rem_y, rem_x)

            block_num = self._calc_block_num(md5[:44])
            if block_num <= 0:
                # print(f'md5:{md5[:44]}没有图片数据')
                x40ImageBigCell = X40ImageBigCell(np.empty(0), md5[:44], smearNo, big_cell_infos, photoType)
            else:
                image_data = self._query_image(block_num, md5[:44])
                if image_data == b'':
                    # print(f'md5:{md5[:44]}获取图片数据失败')
                    x40ImageBigCell = X40ImageBigCell(np.empty(0), md5[:44], smearNo, big_cell_infos, photoType)
                else:
                    x40ImageBigCell = X40ImageBigCell(self._blob_to_numpy(image_data), md5[:44], smearNo,
                                                      big_cell_infos, photoType)

            yield x40ImageBigCell

    def _query_x100_cell_infos(self):
        raise NotImplementedError

    # 获取最大行号
    def get_max_row(self):
        self._cursor.execute(
            'SELECT SmearNo, MAX(YPos) '
            'FROM project_local_tif_pic_info '
            'GROUP BY SmearNo'
        )
        rows = self._cursor.fetchall()
        if len(rows) <= 0:
            return None
        else:
            max_ypos_dict = {s_no: max_y for s_no, max_y in rows}
            return max_ypos_dict

    def get_x40_by_x100(self, x100_image_md5_list):
        max_ypos_dict = self.get_max_row()
        if max_ypos_dict is None:
            return None

        for md5 in x100_image_md5_list:
            md5, smearNo, m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h = md5
            # 获取采样图上的细胞信息
            instance_list = self._query_x100_cell_infos(md5)
            md5 = md5[:44]
            # print(md5)
            # 计算x100采样图对应在分布图上的矩形框四个点坐标，并找出所涉及的所有平扫图
            LU_x = m_upixel_x
            LU_y = m_upixel_y
            LD_x = m_upixel_x
            LD_y = m_upixel_y + m_upixel_h
            RU_x = m_upixel_x + m_upixel_w
            RU_y = m_upixel_y
            RD_x = m_upixel_x + m_upixel_w
            RD_y = m_upixel_y + m_upixel_h
            # 找出左上点所在采样图
            col_LU = int(LU_x / 2448)
            row_LU = int(LU_y / 2048)
            row_LU = max_ypos_dict.get(smearNo) - row_LU
            md5_1 = self.get_x40_image_by_row_col(smearNo, row_LU, col_LU)
            # 找出左下点所在采样图
            col_LD = int(LD_x / 2448)
            row_LD = int(LD_y / 2048)
            row_LD = max_ypos_dict.get(smearNo) - row_LD
            md5_2 = self.get_x40_image_by_row_col(smearNo, row_LD, col_LD)
            # 找出右上点所在采样图
            col_RU = int(RU_x / 2448)
            row_RU = int(RU_y / 2048)
            row_RU = max_ypos_dict.get(smearNo) - row_RU
            md5_3 = self.get_x40_image_by_row_col(smearNo, row_RU, col_RU)
            # 找出右下点所在采样图
            col_RD = int(RD_x / 2448)
            row_RD = int(RD_y / 2048)
            row_RD = max_ypos_dict.get(smearNo) - row_RD
            md5_4 = self.get_x40_image_by_row_col(smearNo, row_RD, col_RD)
            tiles = [
                (row_LU, col_LU, md5_1),
                (row_LD, col_LD, md5_2),
                (row_RU, col_RU, md5_3),
                (row_RD, col_RD, md5_4),
            ]

            # 去重
            unique_tiles = {}
            for row, col, md5X40 in tiles:
                if md5X40 not in unique_tiles:
                    unique_tiles[md5X40] = (row, col, md5X40)

            tiles_unique = list(unique_tiles.values())

            TILE_W = 2448
            TILE_H = 2048
            rows = [row for row, col, _ in tiles_unique]
            cols = [col for _, col, _ in tiles_unique]

            min_col = min(cols)
            max_row = max(rows)

            tiles_norm = []
            for row, col, md5X40 in tiles_unique:
                new_row = max_row - row  # 行翻转：左下为 0
                new_col = col - min_col  # 列归一化
                tiles_norm.append((new_row, new_col, md5X40))

            tiles_norm.sort(key=lambda x: (x[0], x[1]))

            grid_rows = max(r for r, _, _ in tiles_norm) + 1
            grid_cols = max(c for _, c, _ in tiles_norm) + 1

            canvas = np.zeros(
                (grid_rows * TILE_H, grid_cols * TILE_W, 3),
                dtype=np.uint8
            )
            for row, col, md5X40 in tiles_norm:
                md5X40 = md5X40[:44]
                block_num = self._calc_block_num(md5X40)
                image_data = self._query_image(block_num, md5X40)
                if image_data == b'':
                    # print(f'md5:{md5X40}获取图片数据失败')
                    continue

                y0 = row * TILE_H
                y1 = (row + 1) * TILE_H
                x0 = col * TILE_W
                x1 = (col + 1) * TILE_W

                canvas[y0:y1, x0:x1] = self._blob_to_numpy(image_data)

                x_lu = m_upixel_x % TILE_W
                y_lu = m_upixel_y % TILE_H

            image_x40 = canvas[y_lu:y_lu + m_upixel_h, x_lu:x_lu + m_upixel_w]

            block_num = self._calc_block_num(md5)
            if block_num <= 0:
                # print(f'md5:{md5}没有图片数据')
                imageX100 = ImageX100AndX40(np.empty(0), np.empty(0), md5, smearNo, [])
                # imageX100_list.append(imageX100)
                # continue
            else:
                image_data = self._query_image(block_num, md5)
                if image_data == b'':
                    # print(f'md5:{md5}获取图片数据失败')
                    imageX100 = ImageX100AndX40(np.empty(0), np.empty(0), md5, smearNo, [])
                    # imageX100_list.append(imageX100)
                    # continue
                else:
                    # 将BLOB数据转换为 NumPy 数组
                    # np_array = np.frombuffer(image_data, dtype=np.uint8)
                    imageX100 = ImageX100AndX40(self._blob_to_numpy(image_data), image_x40, md5, smearNo, instance_list)
                    # imageX100_list.append(imageX100)
            yield imageX100

    # 获取玻片类型（0-无值， 1-骨髓， 2-血液， 3-尿液）
    def get_smear_type(self):
        # 获取数据库中所有表名
        self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row[0] for row in self._cursor.fetchall()]

        # 编译正则：匹配 project_local_smear_info 或 project_local_smear_info_V<数字>
        pattern = re.compile(r'^project_local_smear_info(?:_V(\d+))?$')

        # 查找匹配的表（正常只有一个）
        target_table = None
        for name in table_names:
            if pattern.match(name):
                target_table = name
                break  # 因为只有一张，找到即可退出

        if target_table is None:
            return {}  # 没有匹配的表，返回空字典

        # 查询该表的数据
        self._cursor.execute(f"SELECT Data, SmearNo FROM {target_table}")
        rows = self._cursor.fetchall()

        # 构建字典
        return {
            smearNo: self._blob_to_u16(data, 142)
            for data, smearNo in rows
        }