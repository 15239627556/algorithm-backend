import cv2
import time
# from algorithms.x40enhance import X40ImageEnhanceModels
# from algorithms.x40model import X40ImageModels

# from algorithms.x100model import X100ImageModels
# from algorithms.x100CSFmodel import X100CSFImageModels

# im1 = cv2.imread("algorithm_source/x100model/1.jpg")
im1 = cv2.imread("algorithm_source/x40model/1.jpg")
# im1 = cv2.imread("algorithm_source/x100CSFmodel/1.jpg")
print(type(im1))
print()
print(im1.shape)
print()
print(im1.dtype)
print()
print(im1.nbytes)
print()
print(im1)
exit()
dispatcher = X40ImageModels.X40ImageModels(num_workers=2)
# dispatcher = X100ImageModels.X100ImageModels(num_workers=2)
# dispatcher = X100CSFImageModels.X100CSFImageModels(num_workers=2)

# dispatcher = X40ImageEnhanceModels.X40ImageEnhanceModels(num_workers=2)
time.sleep(5)
print("开始计时")
start = time.time()
for _ in range(50):
    a = dispatcher.enqueue_task(im1)

dispatcher.synchronize()

end = time.time()
print(f"耗时: {end - start:.6f} 秒")

# del dispatcher
for task_id in range(50):
    result = dispatcher.get_result(task_id)
    # print(result)
    # print("Task ID:", task_id, "result:", result)
    # # print("areaScoreInfo:", result["areaScoreInfo"])
    # # print("bigCellRects:", result["bigCellRects"])
    # # print("haveCellCenterPoints:", result["haveCellCenterPoints"])
    # print("cellRects:", result["areaScoreInfo"])
    # print("cellTypes:", result["haveCellCenterPoints"])
    # print("cellRatios:", result["bigCellRects"])
    # print("enhance_arr", result["enhance_arr"])
    # img = result["enhance_arr"]
    # cv2.imwrite("output.jpg", img)

for i in result["haveCellCenterPoints"]:
    x, y, x1, y1, sore = i
    pt1 = (int(x), int(y))
    pt2 = (int(x1), int(y1))

    cv2.rectangle(im1, pt1, pt2, color=(0, 255, 0), thickness=2)
cv2.imwrite("output.jpg", im1)

# # x40 = X40Main()
# # imgs = [im1, im1, im1, im1]
# # x40.add_x40_task(imgs)
