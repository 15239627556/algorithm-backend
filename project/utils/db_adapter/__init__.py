from .db_adapter import *
from .db_new import ZWXKProjectDBSon0
from .db_old import ZWXKProjectDBSon1

def connect_project(db_path):
    conn, cursor = connect_db(db_path)
    # 判断db版本
    sonClassNO = check_db_versions(cursor)
    zwxkProjectDbSon = None
    if sonClassNO == 0:
        zwxkProjectDbSon = ZWXKProjectDBSon0(cursor)
    elif sonClassNO == 1:
        zwxkProjectDbSon = ZWXKProjectDBSon1(cursor)
    else:
        raise ValueError("未知版本")
    return zwxkProjectDbSon