#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <boost/interprocess/managed_shared_memory.hpp>
#include "common.hpp"
#include "x100Task.hpp"

namespace bip = boost::interprocess;


void process_batch(WorkerSharedBuffer* buffer, WorkerDataBlock &block, int slot_id, X100Main *uInfo, int worker_id) {
    std::vector<cv::Mat> inputImages;
    TaskTypes task_type;
    // 需要根据 image 修改掉 task.result      
    for (int i = 0; i < BATCH_SIZE; ++i) {
        TaskDataBlock &task = block.task_batch_[i];
        printf("Task type: %d\n", task.task_type);
        if (!task.data_filled) continue;
        /* 获取图片 */
        // cv::Mat image(task.image_height, task.image_width, CV_8UC3, task.data);
        int W = 2448;
        int H = 2048;
        if(task.task_type == X100HAVECELL)
        {
            W = 2048;
            H = 1536;
        }
        cv::Mat image(H, W, CV_8UC3, task.data);
        inputImages.push_back(image);
        // /* 计算 */
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));

        /* 写入 task.result */
        task.result.value = task.task_id;
        task_type = task.task_type;
        // printf("Processing task ID: %d, image size: %dx%d\n", task.task_id, task.image_width, task.image_height);
    }
    // uInfo->Location40X(inputImages, block.task_batch_);
    // std::cout << "add_x40_task" << std::endl;
    if (inputImages.size() >= 0 && inputImages.size() <= BATCH_SIZE) {
        uInfo->add_x40_task(inputImages, buffer, slot_id, task_type);
        for (int i = 0; i < BATCH_SIZE; ++i) {
            TaskDataBlock &task = block.task_batch_[i];
            if (!task.data_filled) continue;
            LOGF("Worker Process %d added task %d", worker_id, task.task_id);
        }
    }
        
}
