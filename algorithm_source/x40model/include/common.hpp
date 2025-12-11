#pragma once
#include <opencv2/opencv.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <ctime>
#include "gpu_info.h"

#define LOGF(fmt, ...)                                                                 \
    do {                                                                               \
        using namespace std::chrono;                                                   \
        auto now = system_clock::now();                                                \
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;          \
        std::time_t t = system_clock::to_time_t(now);                                  \
        std::tm tm = *std::localtime(&t);                                              \
        char time_buf[64];                                                             \
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm);           \
        printf("[%s.%03lld] " fmt "\n", time_buf, (long long)ms.count(), ##__VA_ARGS__); \
    } while (0)

namespace py = pybind11;

#define WORKER_QUEUE_SIZE   10
#define BATCH_SIZE          4
#define IMAGE_MAX_W         2448
#define IMAGE_MAX_H         2048
#define HAVE_CELL_NUM_MAX   2000
#define BIG_CELL_NUM_MAX    10
#define BLOCK_NUM           4

namespace bip = boost::interprocess;

enum TaskStatus {
    FILLING   = 0,      /* 正在填充图片 (Batch Size 未满) */
    READY     = 1,      /* 图片填充完毕，需要计算 */
    DONE      = 2,      /* 计算完毕，结果已准备好 */
    COMPUTING = 3,      /* 正在计算 */
};

struct Task {
    cv::Mat image;
    int task_id;
};

/*采样图上的模型计算结果-细胞举行框信息*/
struct cellRectInfo{
    int                             x;
    int                             y;
    int                             w;
    int                             h;
    float                           rate;
};

/*采样图上的模型计算结果*/
struct HaveCellResultInfos{
    int                             x1;
    int                             y1;
    int                             x2;
    int                             y2;
    float                           score;
};
/*采样图上的区域分值*/
struct AreaScoreInfo{
    int                             x;//该区域在采样图的左上点x
    int                             y;//该区域在采样图的左上点y
    int                             w;//该区域宽
    int                             h;//该区域高
    float                           score;//该区域分值
    int                             grade;//该区域等级
};

/*采样图上的模型计算结果信息*/
struct ImageResultInfos{
    int                             bigCellRectsSize;// 单张采样图上巨核细胞实际数量               
    cellRectInfo                    bigCellRects[BIG_CELL_NUM_MAX]; //单张采样图上所有巨核细胞框
    int                             haveCellCenterPointsSize;//单张采样图上有核细胞中心点实际数量
    HaveCellResultInfos             haveCellCenterPoints[HAVE_CELL_NUM_MAX]; //单张采样图上所有有核细胞中心点
    AreaScoreInfo                   areaScoreInfo[BLOCK_NUM];//采样图上四宫格信息
};

struct TaskResult {
    int                             value;
    ImageResultInfos                imageResultInfos;
   // int                             actualSize; //imageResultInfos实际长度（有时候可能不满四张图）
   // ImageResultInfos                imageResultInfos[BATCH_SIZE];
};

typedef py::dict pyTaskResultType;
// const pyTaskResultType py_task_no_result = py::dict{};

pyTaskResultType convert_to_py_task_result(const TaskResult &result){
    
    // constexpr ssize_t B = BATCH_SIZE;
    const ssize_t N1 = result.imageResultInfos.bigCellRectsSize;
    const ssize_t N2 = result.imageResultInfos.haveCellCenterPointsSize;
    constexpr ssize_t N3 = BLOCK_NUM;

    py::dict output;
    output["value"] = result.value;

    std::vector<ssize_t> rect_shape = {N1, 5};
    py::array_t<float> big_rects(rect_shape);
    auto rects_buf = big_rects.mutable_unchecked<2>();

    std::vector<ssize_t> center_shape = {N2, 5};
    py::array_t<float> cell_centers(center_shape);
    auto centers_buf = cell_centers.mutable_unchecked<2>();

    std::vector<ssize_t> score_shape = {N3, 6};
    py::array_t<float> area_scores(score_shape);
    auto score_buf = area_scores.mutable_unchecked<2>();

    const auto& img = result.imageResultInfos;

    for (ssize_t i = 0; i < N1; ++i) {
        rects_buf(i, 0) = img.bigCellRects[i].x;
        rects_buf(i, 1) = img.bigCellRects[i].y;
        rects_buf(i, 2) = img.bigCellRects[i].w;
        rects_buf(i, 3) = img.bigCellRects[i].h;
        rects_buf(i, 4) = img.bigCellRects[i].rate;
    }

    for (ssize_t i = 0; i < N2; ++i) {
        centers_buf(i, 0) = img.haveCellCenterPoints[i].x1;
        centers_buf(i, 1) = img.haveCellCenterPoints[i].y1;
        centers_buf(i, 2) = img.haveCellCenterPoints[i].x2;
        centers_buf(i, 3) = img.haveCellCenterPoints[i].y2;
        centers_buf(i, 4) = img.haveCellCenterPoints[i].score;
    }

    for (ssize_t i = 0; i < N3; ++i) {
        score_buf(i, 0) = img.areaScoreInfo[i].x;
        score_buf(i, 1) = img.areaScoreInfo[i].y;
        score_buf(i, 2) = img.areaScoreInfo[i].w;
        score_buf(i, 3) = img.areaScoreInfo[i].h;
        score_buf(i, 4) = img.areaScoreInfo[i].score;
        score_buf(i, 5) = img.areaScoreInfo[i].grade;
    }

    // for (ssize_t b = 0; b < B; ++b) {
    //     const auto& img = result.imageResultInfos;

    //     for (ssize_t i = 0; i < N1; ++i) {
    //         rects_buf(b, i, 0) = img.bigCellRects[i].x;
    //         rects_buf(b, i, 1) = img.bigCellRects[i].y;
    //         rects_buf(b, i, 2) = img.bigCellRects[i].w;
    //         rects_buf(b, i, 3) = img.bigCellRects[i].h;
    //     }

    //     for (ssize_t i = 0; i < N2; ++i) {
    //         centers_buf(b, i, 0) = img.haveCellCenterPoints[i].c_x;
    //         centers_buf(b, i, 1) = img.haveCellCenterPoints[i].c_y;
    //     }

    //     for (ssize_t i = 0; i < N3; ++i) {
    //         score_buf(b, i, 0) = img.areaScoreInfo[i].x;
    //         score_buf(b, i, 1) = img.areaScoreInfo[i].y;
    //         score_buf(b, i, 2) = img.areaScoreInfo[i].w;
    //         score_buf(b, i, 3) = img.areaScoreInfo[i].h;
    //         score_buf(b, i, 4) = img.areaScoreInfo[i].score;
    //     }
    // }

    output["bigCellRects"] = big_rects;
    output["haveCellCenterPoints"] = cell_centers;
    output["areaScoreInfo"] = area_scores;

    // output["bigCellRectsSize"] = result.imageResultInfos.bigCellRectsSize;
    // output["haveCellCenterPointsSize"] = result.imageResultInfos.haveCellCenterPointsSize;
    // return result.value;
    return output;
}


struct TaskDataBlock {
    int                             task_id;
    int                             image_width;
    int                             image_height;
    bool                            data_filled = false;
    unsigned char                   data[IMAGE_MAX_W * IMAGE_MAX_H * 3];
    TaskResult                      result;
};

struct WorkerDataBlock {
    TaskDataBlock                   task_batch_[BATCH_SIZE];
    TaskStatus                      task_status = FILLING;
    bip::interprocess_mutex         mutex_block_data_;
    int next_batch_index() {
        bip::scoped_lock<bip::interprocess_mutex> lock(mutex_block_data_);
        for (int i = 0; i < BATCH_SIZE; ++i) {
            if (!task_batch_[i].data_filled) return i;
        }
        return -1;
    }
};

struct WorkerSharedBuffer {
    WorkerDataBlock                 blocks[WORKER_QUEUE_SIZE];
    bip::interprocess_condition     cv_new_task_;
    bip::interprocess_condition     cv_result_ready_;
    bip::interprocess_condition     cv_slot_available_;
    bip::interprocess_mutex         mutex_buffer_;
    bool                            stop_flag_ = false;
    int next_available() {
        for (int i = 0; i < WORKER_QUEUE_SIZE; ++i) {
            if (blocks[i].task_status == FILLING) return i;
        }
        return -1;
    }
    int next_ready() {
        for (int i = 0; i < WORKER_QUEUE_SIZE; ++i) {
            if (blocks[i].task_status == READY) return i;
        }
        return -1;
    }
    int next_done() {
        for (int i = 0; i < WORKER_QUEUE_SIZE; ++i) {
            if (blocks[i].task_status == DONE) return i;
        }
        return -1;
    }
    int wait_next_ready_block() {
        int slot_id = -1;
        bip::scoped_lock<bip::interprocess_mutex> cv_lock(mutex_buffer_);
        cv_new_task_.wait(cv_lock, [&] {
            slot_id = next_ready();
            return slot_id != -1;
        });
        return slot_id;
    }
    int wait_next_done_block() {
        int slot_id = -1;
        bip::scoped_lock<bip::interprocess_mutex> cv_lock(mutex_buffer_);
        cv_result_ready_.wait(cv_lock, [&] {
            slot_id = next_done();
            return (slot_id != -1) || stop_flag_;
        });
        return slot_id;
    }
    int wait_next_available_block() {
        int slot_id = next_available();
        if (slot_id == -1) {
            bip::scoped_lock<bip::interprocess_mutex> cv_lock(mutex_buffer_);
            cv_slot_available_.wait(cv_lock, [&] {
                slot_id = next_available();
                return slot_id != -1;
            });
        }
        return slot_id;
    }
    void upload_task(Task &task, int slot_id) {
        int batch_index = blocks[slot_id].next_batch_index();
        if (batch_index == -1) {
            throw std::runtime_error("No available batch slot in the worker data block.");
        }
        // printf("Uploading task ID: %d to slot ID: %d, batch index: %d\n", task.task_id, slot_id, batch_index);
        bip::scoped_lock<bip::interprocess_mutex> lock(blocks[slot_id].mutex_block_data_);
        blocks[slot_id].task_batch_[batch_index].task_id = task.task_id;
        blocks[slot_id].task_batch_[batch_index].image_width = task.image.cols;
        blocks[slot_id].task_batch_[batch_index].image_height = task.image.rows;
        blocks[slot_id].task_batch_[batch_index].data_filled = true;
        std::memcpy(blocks[slot_id].task_batch_[batch_index].data, task.image.data, task.image.cols * task.image.rows * 3);
        if (batch_index == BATCH_SIZE - 1) {
            bip::scoped_lock<bip::interprocess_mutex> cv_lock(mutex_buffer_);
            blocks[slot_id].task_status = READY;
            cv_new_task_.notify_all();
        }
    }
    void force_ready() {
        for (int i = 0; i < WORKER_QUEUE_SIZE; ++i) {
            {
                bip::scoped_lock<bip::interprocess_mutex> cv_lock(mutex_buffer_);
                WorkerDataBlock &block = blocks[i];
                if (block.task_status == FILLING) {
                    for (int j = 0; j < BATCH_SIZE; ++j) {
                        if (block.task_batch_[j].data_filled) {
                            block.task_status = READY;
                            continue;
                        } else {
                            break;
                        }
                    }
                }
                cv_new_task_.notify_all();
            }
        }
    }
    bool all_filling() {
        for (int i = 0; i < WORKER_QUEUE_SIZE; ++i) {
            bip::scoped_lock<bip::interprocess_mutex> cv_lock(mutex_buffer_);
            if (blocks[i].task_status != FILLING) return false;
        }
        return true;
    }
};
