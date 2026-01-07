from db_parent import ZWXKProjectDB, ImageX100
import numpy as np


class ZWXKProjectDBSon0(ZWXKProjectDB):
    def __init__(self, db_cursor):
        super().__init__(db_cursor)

    # 获取巨核细胞所有采样图的MD5以及在分布图位置
    def get_big_cell_image_infos(self):
        big_cell_image_list = []
        tableNames = [
            "project_local_pic_info_V1",
            "project_local_pic_info_V2",
            "project_local_pic_info_V3",
            "project_local_pic_info_V4",
            "project_local_pic_info_V5"
        ]
        for name in tableNames:
            self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
            if self._cursor.fetchone() is None:
                continue
            self._cursor.execute(
                f'SELECT FileId, SmearNo, Data, PhotoType FROM {name} '
                'WHERE CollectObject = ?',
                (1, ))
            infos = self._cursor.fetchall()
            if len(infos) > 0:
                for i_index, i in enumerate(infos):
                    md5 = i[0]
                    smearNo = i[1]
                    data = i[2]
                    m_upixel_x = self._blob_to_u32(data, 18)  # 采样图在分布图上位置x 左上角是分布图(0, 0)点
                    m_upixel_y = self._blob_to_u32(data, 22)  # 采样图在分布图上位置y 左上角是分布图(0, 0)点
                    m_upixel_w = self._blob_to_u32(data, 26)  # 采样图在分布图上位置w 左上角是分布图(0, 0)点
                    m_upixel_h = self._blob_to_u32(data, 30)  # 采样图在分布图上位置h 左上角是分布图(0, 0)点
                    image_w = self._blob_to_u16(data, 14)
                    image_h = self._blob_to_u16(data, 16)
                    photoType = i[3] # 5-从分布图直接截图的巨核 6-从40x转百倍拍摄的巨核

                    big_cell_image_list.append([md5, smearNo, m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h, image_w, image_h, photoType])
                break
        return big_cell_image_list

    # 查询所有玻片的x100模式拍摄的有核采样图md5(未截取有效数据的md5)
    def _query_x100_images_md5(self):
        x100_image_md5_list = []
        tableNames = [
            "project_local_pic_info_V1",
            "project_local_pic_info_V2",
            "project_local_pic_info_V3",
            "project_local_pic_info_V4",
            "project_local_pic_info_V5"
        ]
        for name in tableNames:
            self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
            if self._cursor.fetchone() is None:
                continue
            self._cursor.execute(
                f'SELECT FileId, SmearNo, Data FROM {name} '
                'WHERE PhotoType = ?  AND CollectObject = ?',
                (6, 0))
            infos = self._cursor.fetchall()
            if len(infos) > 0:
                for i_index, i in enumerate(infos):
                    md5, smearNo, data = i
                    m_upixel_x = self._blob_to_u32(data, 18)  # 采样图在分布图上位置x
                    m_upixel_y = self._blob_to_u32(data, 22)  # 采样图在分布图上位置y
                    m_upixel_w = self._blob_to_u32(data, 26)  # 采样图在分布图上位置w
                    m_upixel_h = self._blob_to_u32(data, 30)  # 采样图在分布图上位置h
                    x100_image_md5_list.append([md5, smearNo, m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h])
                break
        return x100_image_md5_list

    # 查询所有玻片的x100模式拍摄的巨核采样图md5(未截取有效数据的md5)
    def _query_x100_meg_images_md5(self):
        x100_image_meg_md5_list = []
        tableNames = [
            "project_local_pic_info_V1",
            "project_local_pic_info_V2",
            "project_local_pic_info_V3",
            "project_local_pic_info_V4",
            "project_local_pic_info_V5"
        ]
        for name in tableNames:
            self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
            if self._cursor.fetchone() is None:
                continue
            self._cursor.execute(
                f'SELECT FileId, SmearNo, Data FROM {name} '
                'WHERE PhotoType = ?  AND CollectObject = ?',
                (6, 1))
            infos = self._cursor.fetchall()
            if len(infos) > 0:
                for i_index, i in enumerate(infos):
                    md5, smearNo, data = i
                    m_upixel_x = self._blob_to_u32(data, 18)  # 采样图在分布图上位置x
                    m_upixel_y = self._blob_to_u32(data, 22)  # 采样图在分布图上位置y
                    m_upixel_w = self._blob_to_u32(data, 26)  # 采样图在分布图上位置w
                    m_upixel_h = self._blob_to_u32(data, 30)  # 采样图在分布图上位置h
                    x100_image_meg_md5_list.append([md5, smearNo, m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h])
                break
        return x100_image_meg_md5_list

    # 查询巨核细胞信息
    def _query_big_cell_infos(self, md5,  w_ratio, h_ratio, rem_y, rem_x):
        # check:审核端下载的db, watch:观察端下载的db, report:报告端下载的db
        tableNames = [
            "project_local_doc_cell_info_check_V1",
            "project_local_doc_cell_info_check_V2",
            "project_local_doc_cell_info_check_V3",
            "project_local_doc_cell_info_watch_V1",
            "project_local_doc_cell_info_watch_V2",
            "project_local_doc_cell_info_watch_V3",
            "project_local_doc_cell_info_report_V1",
            "project_local_doc_cell_info_report_V2",
            "project_local_doc_cell_info_report_V3"
        ]
        big_cell_info_list = []

        for name in tableNames:
            self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
            if self._cursor.fetchone() is None:
                continue
            self._cursor.execute(
                f'SELECT XPos, YPos, XLength, YLength, Data FROM {name} '
                f'WHERE FileId = ? AND DoctorType = ?',
                (md5, 3))

            infos = self._cursor.fetchall()
            if len(infos) <= 0:  # 没有审核人数据,拿报告人数据
                # print("_query_x100_cell_infos 不存在审核人数据")
                self._cursor.execute(
                    f'SELECT XPos, YPos, XLength, YLength, Data FROM {name} '
                    f'WHERE FileId = ? AND DoctorType = ?',
                    (md5, 1))
                infos = self._cursor.fetchall()
            if len(infos) > 0:
                for i_index, i in enumerate(infos):
                    x = i[0]
                    y = i[1]
                    w = i[2]
                    h = i[3]
                    data = i[4]
                    m_usame_exist = self._blob_to_u8(data, 45)
                    m_umodify_type = self._blob_to_u32(data, 0)
                    if m_usame_exist != 1 and m_umodify_type not in [318]:  # 318-未使用细胞，283-带鉴定细胞
                        m_ucell_type1 = self._blob_to_u32(data, 4)
                        big_cell_info_list.append([x * w_ratio + rem_x, y * h_ratio + rem_y, w * w_ratio, h * h_ratio, m_umodify_type, m_ucell_type1])
                break
        return big_cell_info_list

    # 查询指定采样图的有效有核细胞信息列表(默认查询审核医生数据，若不存在审核医生数据则查询报告医生数据)
    def _query_x100_cell_infos(self, md5):
        # check:审核端下载的db, watch:观察端下载的db, report:报告端下载的db
        tableNames = [
            "project_local_doc_cell_info_check_V1",
            "project_local_doc_cell_info_check_V2",
            "project_local_doc_cell_info_check_V3",
            "project_local_doc_cell_info_watch_V1",
            "project_local_doc_cell_info_watch_V2",
            "project_local_doc_cell_info_watch_V3",
            "project_local_doc_cell_info_report_V1",
            "project_local_doc_cell_info_report_V2",
            "project_local_doc_cell_info_report_V3"
        ]
        x100_cell_info_list = []
        for name in tableNames:
            self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
            if self._cursor.fetchone() is None:
                continue
            # self._cursor.execute(
            #     f'SELECT XPos, YPos, XLength, YLength, Data FROM {name} '
            #     f'WHERE FileId = ? AND DoctorType = ? AND States = ?',
            #     (md5, 3, 0))
            if self.column_exists(name, "States"):
                sql = (
                    f"SELECT XPos, YPos, XLength, YLength, Data "
                    f"FROM {name} "
                    f"WHERE FileId = ? AND DoctorType = ? AND States = ?"
                )
                params = (md5, 3, 0)
            else:
                sql = (
                    f"SELECT XPos, YPos, XLength, YLength, Data "
                    f"FROM {name} "
                    f"WHERE FileId = ? AND DoctorType = ?"
                )
                params = (md5, 3)

            self._cursor.execute(sql, params)

            infos = self._cursor.fetchall()
            if len(infos) <= 0:  # 没有审核人数据,拿报告人数据
                # print("_query_x100_cell_infos 不存在审核人数据")
                self._cursor.execute(
                    f'SELECT XPos, YPos, XLength, YLength, Data FROM {name} '
                    f'WHERE FileId = ? AND DoctorType = ? AND States = ?',
                    (md5, 1, 0))
                infos = self._cursor.fetchall()
            if len(infos) > 0:
                for i_index, i in enumerate(infos):
                    x = i[0]
                    y = i[1]
                    w = i[2]
                    h = i[3]
                    # md5 = i[4]
                    # # 截取md5
                    # md5 = md5[:44]
                    data = i[4]
                    m_usame_exist = self._blob_to_u8(data, 45)
                    m_umodify_type = self._blob_to_u32(data, 0)
                    if m_usame_exist != 1 and m_umodify_type not in [318]:  # 318-未使用细胞，283-带鉴定细胞
                        m_ucell_type1 = self._blob_to_u32(data, 4)
                        x100_cell_info_list.append([x, y, w, h, m_umodify_type, m_ucell_type1])
                break
        return x100_cell_info_list

    # 获取所有玻片的所有指定层40x平扫图
    def get_x40_sample_images(self, keep_ratio=1):
        return self.get_x40_sample_images_()

    # 获取所有玻片的所有x100采样图及细胞信息（有核）
    def get_x100_sample_images(self):
        x100_image_md5_list = self._query_x100_images_md5()
        # 获取采样图上的细胞信息
        for md5 in x100_image_md5_list:
            md5, smearNo, m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h = md5
            instance_list = self._query_x100_cell_infos(md5)
            md5 = md5[:44]
            block_num = self._calc_block_num(md5)
            if block_num <= 0:
                # print(f'md5:{md5}没有图片数据')
                imageX100 = ImageX100(np.empty(0), md5, smearNo, instance_list,
                                      m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h)
            else:
                image_data = self._query_image(block_num, md5)
                if image_data == b'':
                    # print(f'md5:{md5}获取图片数据失败')
                    imageX100 = ImageX100(np.empty(0), md5, smearNo, instance_list,
                                          m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h)
                else:
                    imageX100 = ImageX100(self._blob_to_numpy(image_data), md5, smearNo, instance_list,
                                          m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h)
            yield imageX100

    # 获取所有玻片的所有x100采样图及细胞信息（巨核）
    def get_x100_meg_images(self):
        x100_image_meg_md5_list = self._query_x100_meg_images_md5()

        # 获取采样图上的细胞信息
        for md5 in x100_image_meg_md5_list:
            md5, smearNo, m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h = md5
            instance_list = self._query_x100_cell_infos(md5)
            md5 = md5[:44]
            block_num = self._calc_block_num(md5)
            if block_num <= 0:
                # print(f'md5:{md5}没有图片数据')
                imageX100Meg = ImageX100(np.empty(0), md5, smearNo, instance_list, 0, 0, 0, 0)
            else:
                image_data = self._query_image(block_num, md5)
                if image_data == b'':
                    # print(f'md5:{md5}获取图片数据失败')
                    imageX100Meg = ImageX100(np.empty(0), md5, smearNo, instance_list, 0, 0, 0, 0)
                else:
                    imageX100Meg = ImageX100(self._blob_to_numpy(image_data), md5, smearNo, instance_list, m_upixel_x, m_upixel_y, m_upixel_w, m_upixel_h)
            yield imageX100Meg

    # 获取x100有核像素为米转换关系
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
        x100_image_md5_list = self._query_x100_images_md5()
        return self.get_x40_by_x100(x100_image_md5_list)

    # 获取x100巨核采样图以及其所在分布图的区域截图
    def get_x100_big_and_x40Area(self):
        x100_big_image_ms5_list = self.get_big_cell_image_infos()
        return self.get_x40_by_x100(x100_big_image_ms5_list)