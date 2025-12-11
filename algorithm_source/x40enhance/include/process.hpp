#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <boost/interprocess/managed_shared_memory.hpp>
#include "common.hpp"
#include "x40Task3.hpp"

namespace bip = boost::interprocess;


void process_batch(WorkerSharedBuffer* buffer, WorkerDataBlock &block, int slot_id, X40Enhance *uInfo, int worker_id) {
    std::vector<cv::Mat> inputImages;
    // 需要根据 image 修改掉 task.result      
    for (int i = 0; i < BATCH_SIZE; ++i) {
        TaskDataBlock &task = block.task_batch_[i];
        if (!task.data_filled) continue;
        /* 获取图片 */
        cv::Mat image(task.image_height, task.image_width, CV_8UC3, task.data);
        inputImages.push_back(image);
        // /* 计算 */
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));

        /* 写入 task.result */
        task.result.value = task.task_id;
        printf("Processing task ID: %d, image size: %dx%d\n", task.task_id, task.image_width, task.image_height);
    }
    // uInfo->Location40X(inputImages, block.task_batch_);
    // std::cout << "add_x40_task" << std::endl;
    if (inputImages.size() == BATCH_SIZE) {
        uInfo->add_x40_enhance_task(inputImages[0], buffer, slot_id);
        for (int i = 0; i < BATCH_SIZE; ++i) {
            std::cout << BATCH_SIZE << std::endl;
            TaskDataBlock &task = block.task_batch_[i];
            if (!task.data_filled) continue;
            LOGF("Worker Process %d added task %d", worker_id, task.task_id);
        }
    }
        
}
