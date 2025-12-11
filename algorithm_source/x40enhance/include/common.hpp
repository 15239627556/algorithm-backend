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

#define WORKER_QUEUE_SIZE   5
#define BATCH_SIZE          1
#define IMAGE_MAX_W         2448
#define IMAGE_MAX_H         2048

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

struct TaskResult {
    int                             value;
    unsigned char                   data[IMAGE_MAX_W * IMAGE_MAX_H * 3];
};


typedef py::dict pyTaskResultType;
// const pyTaskResultType py_task_no_result = py::dict{};

pyTaskResultType convert_to_py_task_result(const TaskResult &r){
    
    const size_t n = static_cast<size_t>(IMAGE_MAX_W) * IMAGE_MAX_H * 3;
    if (n > sizeof(r.data)) throw std::runtime_error("w*h*3 exceeds buffer");

    // 创建一个 HxWx3 的 uint8 NumPy 数组，并把数据拷进去
    py::array_t<uint8_t> arr({IMAGE_MAX_H, IMAGE_MAX_W, 3});
    std::memcpy(arr.mutable_data(), r.data, n);

    // std::cout << "r.value====> " << r.value << std::endl;
    // std::cout << "r.value====> " << r.data << std::endl;

    py::dict d;
    d["value"] = r.value;
    d["enhance_arr"]  = std::move(arr);         // d["data"] 是 numpy.ndarray
    return d;
}


struct TaskDataBlock {
    int                             task_id;
    int                             image_width;
    int                             image_height;
    bool                            data_filled = false;
    unsigned char                   data[IMAGE_MAX_W * IMAGE_MAX_H * 3];
    // unsigned char                   data_[IMAGE_MAX_W * IMAGE_MAX_H * 3];
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
