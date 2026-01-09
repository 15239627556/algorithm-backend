from .db_parent import ZWXKProjectDB, ImageX100
import numpy as np


class ZWXKProjectDBSon1(ZWXKProjectDB):
    def __init__(self, db_cursor):
        super().__init__(db_cursor)

    # 查询所有玻片的x100模式拍摄的采样图md5(未截取有效数据的md5)
    def _query_x100_images_md5(self):
        x100_image_md5_list = []
        # 此版project_local_pic_info表结构中不包含有核巨核采集的区分，应该是此版本没有x100巨核采样模式.Section_Seq拨片的序列号是对应的玻片编号
        self._cursor.execute(
            'SELECT FileId, SectionSeq, Data FROM project_local_pic_info '
            'WHERE PhotoType = ?',
            (6,))
        infos = self._cursor.fetchall()
        for i_index, i in enumerate(infos):
            md5 = i[0]
            # # 截取md5
            # md5 = md5[:44]
            smearNo = i[1]
            data = i[2]
            m_upixel_x = self._blob_to_u32(data, 18)  # 采样图在分布图上位置x
            m_upixel_y = self._blob_to_u32(data, 22)  # 采样图在分布图上位置y
            m_upixel_w = self._blob_to_u32(data, 26)  # 采样图在分布图上位置w
            m_upixel_h = self._blob_to_u32(data, 30)  # 采样图在分布图上位置h
            x100_image_md5_list.append([md5, smearNo, m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h])
        return x100_image_md5_list

    # 查询指定采样图的有效有核细胞信息列表(默认查询审核医生数据，若不存在审核医生数据则查询报告医生数据)
    def _query_x100_cell_infos(self, md5):
        # print("旧版本，导出信息可能不准确，需要根据db表格确认")
        x100_cell_info_list = []
        column_exists_flag = self.column_exists("project_local_doc_cell_info", "States")
        if column_exists_flag:
            self._cursor.execute(
                f'SELECT XPos, YPos, XLength, YLength, CellType, IsDel FROM project_local_doc_cell_info '
                f'WHERE DoctorType = ? AND FileId = ?  AND States = ?',
                (3, md5, 0))
        else:
            # print("字段 States 不存在")
            self._cursor.execute(
                f'SELECT XPos, YPos, XLength, YLength, CellType, IsDel FROM project_local_doc_cell_info '
                f'WHERE DoctorType = ? AND FileId = ?',
                (3, md5))

        infos = self._cursor.fetchall()
        if len(infos) <= 0:  # 没有审核人数据,拿报告人数据
            # print("_query_x100_cell_infos 不存在审核人数据")
            if column_exists_flag:
                self._cursor.execute(
                    f'SELECT XPos, YPos, XLength, YLength, CellType, IsDel FROM project_local_doc_cell_info '
                    f'WHERE DoctorType = ? AND FileId = ?  AND States = ?',
                    (1, md5, 0))
            else:
                self._cursor.execute(
                    f'SELECT XPos, YPos, XLength, YLength, CellType, IsDel FROM project_local_doc_cell_info '
                    f'WHERE DoctorType = ? AND FileId = ?',
                    (3, md5))
            infos = self._cursor.fetchall()
        for i_index, i in enumerate(infos):
            x = i[0]
            y = i[1]
            w = i[2]
            h = i[3]
            # md5 = i[4]
            # # 截取md5
            # md5 = md5[:44]
            data = i[4]
            m_uis_del = i[5]
            m_umodify_type = self._blob_to_u32(data, 0)
            if m_uis_del != 1 and m_umodify_type not in [318]:  # 318-未使用细胞，283-带鉴定细胞
                m_ucell_type1 = self._blob_to_u32(data, 4)
                x100_cell_info_list.append([x, y, w, h, m_umodify_type, m_ucell_type1])
        return x100_cell_info_list

    # 获取所有玻片的所有指定层40x平扫图
    def get_x40_sample_images(self, keep_ratio=1):
        return self.get_x40_sample_images_(keep_ratio)

    # 获取x100采样图及细胞信息（有核）
    def get_x100_sample_images(self):
        # imageX100_list = []
        x100_image_md5_list = self._query_x100_images_md5()

        # 获取采样图上的细胞信息
        for md5 in x100_image_md5_list:
            smearNo = md5[1]
            instance_list = self._query_x100_cell_infos(md5[0])
            md5 = md5[0][:44]
            block_num = self._calc_block_num(md5)
            if block_num <= 0:
                # print(f'md5:{md5}没有图片数据')
                imageX100 = ImageX100(np.empty(0), md5, smearNo, instance_list)
                # imageX100_list.append(imageX100)
                # continue
            else:
                image_data = self._query_image(block_num, md5)
                if image_data == b'':
                    # print(f'md5:{md5}获取图片数据失败')
                    imageX100 = ImageX100(np.empty(0), md5, smearNo, instance_list)
                    # imageX100_list.append(imageX100)
                    # continue
                else:
                    # 将BLOB数据转换为 NumPy 数组
                    # np_array = np.frombuffer(image_data, dtype=np.uint8)
                    imageX100 = ImageX100(self._blob_to_numpy(image_data), md5, smearNo, instance_list)
                    # imageX100_list.append(imageX100)
            yield imageX100

    # 获取x100采样图及细胞信息（巨核；此版本没有x100镜头拍摄巨核细胞的功能）
    def get_x100_meg_images(self):
        return None

    #获取x100有核像素为米转换关系
    def get_x100_have_um_px(self):
        box = []
        self._cursor.execute(
            f'SELECT Data, SmearNo, UpLoad FROM project_local_smear_info_V2')
        datas = self._cursor.fetchall()
        if len(datas) > 0:
            for i_index, i in enumerate(datas):
                data = i[0]
                smearNo = i[1]
                upLoad = i[2]
                # if upLoad == 1:
                m_uselect_num = self._blob_to_u32(data, 156)
                m_uX100_px = self._blob_to_u32(data, 160 + m_uselect_num * 17 + 57)
                m_uX100_um = self._blob_to_u32(data, 160 + m_uselect_num * 17 + 61)
                box.append([smearNo, m_uX100_px, m_uX100_um])
        return box

    # 获取x100有核采样图以及其所在分布图的区域截图
    def get_x100_have_and_x40Area(self):
        table_name = "project_local_tif_pic_info"
        self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if self._cursor.fetchone() is None:
            # print(f"不存在表 {table_name}, 应是未下载x40平扫图")
            return []
        x100_image_md5_list = self._query_x100_images_md5()
        return self.get_x40_by_x100(x100_image_md5_list)


