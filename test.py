import cv2
import time
# from algorithms.x40enhance import X40ImageEnhanceModels
from algorithms.x40model import X40ImageModels
# from algorithms.x100model import X100ImageModels
# from algorithms.x100CSFmodel import X100CSFImageModels

# im1 = cv2.imread("algorithm_source/x100model/rbc.png")
im1 = cv2.imread("algorithm_source/x40model/1.jpg")
im2 = cv2.imread("algorithm_source/x40model/2.jpg")
# im1 = cv2.imread("algorithm_source/x100CSFmodel/4.jpg")
# im1 = cv2.imread("algorithm_source/x100CSFmodel/3.jpg")

dispatcher = X40ImageModels.X40ImageModels(num_workers=3)
# dispatcher = X100ImageModels.X100ImageModels(num_workers=2)
# dispatcher = X100CSFImageModels.X100CSFImageModels(num_workers=2)

# dispatcher = X40ImageEnhanceModels.X40ImageEnhanceModels(num_workers=2)
time.sleep(5)
print("开始计时")
start = time.time()
for _ in range(122):
    a = dispatcher.enqueue_task(im1)
    print("enqueue_task:", a)
# for _ in range(500):
#     a = dispatcher.enqueue_task(im2)

dispatcher.synchronize()

end = time.time()
print(f"耗时: {end - start:.6f} 秒")


# del dispatcher
for task_id in range(122):
    result = dispatcher.get_result(task_id)
    if not result:
        print("Task ID:", task_id, "result:", result)
    # print("areaScoreInfo:", result["areaScoreInfo"])
    # print("bigCellRects:", result["bigCellRects"])
    # print("haveCellCenterPoints:", result["haveCellCenterPoints"])
    # print("cellRects:", result["cellRects"])
    # print("cellTypes:", result["cellTypes"])
    # print("cellRatios:", result["cellRatios"])
    # print("enhance_arr", result["enhance_arr"])
    # img = result["enhance_arr"]
    # cv2.imwrite("output.jpg", img)
    # cellnum = 0
    # img = im2.copy()
    # for index, i in enumerate(result["haveCellCenterPoints"]):
    #     x1, y1, x2, y2, score = i
    #     pt1 = (int(x1), int(y1))
    #     pt2 = (int(x2), int(y2))
    #     cellnum += 1
    #
    #     cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=2)
    #
    # for i in result["bigCellRects"]:
    #     x, y, w, h, score = i
    #     pt1 = (int(x), int(y))
    #     pt2 = (int(x + w), int(y + h))
    #     cv2.rectangle(img, pt1, pt2, color=(255, 0, 0), thickness=2)
    #
    #
    # # for index, i in enumerate(result["cellRects"]):
    # #     x, y, w, h = i
    # #     pt1 = (x, y)
    # #     pt2 = (x + w, y + h)
    # #     cellnum += 1
    #
    # #     cv2.rectangle(im1, pt1, pt2, color=(0, 255, 0), thickness=2)
    #
    # #     cv2.putText(im1,
    # #             str(result["cellTypes"][index][0]),
    # #             (x, y -5),                      # 坐标
    # #             cv2.FONT_HERSHEY_SIMPLEX,       # 字体
    # #             1.2,                            # 字体大小
    # #             (0, 255, 0),                    # 颜色：绿色
    # #             2,                              # 粗细
    # #             cv2.LINE_AA)                    # 抗锯齿线条
    #
    # print("cellnum == ", cellnum)
    # cv2.imwrite("output.jpg", img)

# # x40 = X40Main()
# # imgs = [im1, im1, im1, im1]
# # x40.add_x40_task(imgs)




