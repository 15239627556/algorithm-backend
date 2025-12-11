#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <queue>
#include <map>
#include "common.hpp"
#include "workerwrapper.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define ENABLE_LOGGING 1

namespace py = pybind11;

class X40ImageEnhanceModels {

public:

    X40ImageEnhanceModels(int num_workers) {
        for (int i = 0; i < num_workers; ++i) {
            TaskWorkerWrapper *worker = new TaskWorkerWrapper(i);
            workers_.emplace_back(worker);
            worker_task_dispatching_threads_.emplace_back([this, i]() {this->worker_task_dispatching_loop_(i); });
            worker_result_fetching_threads_.emplace_back([this, i]() {this->worker_result_fetching_loop_(i); });
        }
    }

    ~X40ImageEnhanceModels() {
        dispatching_threads_stopped_ = true;
        result_fetching_threads_stopped_ = true;
        cv_new_task_.notify_all();
        for (auto& thread : worker_task_dispatching_threads_) {
            if (thread.joinable()) thread.join();
        }
        for (auto& worker : workers_) {
            worker->get_buffer()->stop_flag_ = true;
            worker->get_buffer()->cv_result_ready_.notify_all();
        }
        for (auto& thread : worker_result_fetching_threads_) {
            if (thread.joinable()) thread.join();
        }
        for (auto& worker : workers_) {
            delete worker;
        }
    }

    /* 往队列添加一张图片 */
    int enqueue_task(py::array image) {
        int task_id;
        // printf("Enqueuing task id: %d\n", task_id_counter_);
        task_id = task_id_counter_++;
        task_id_counter_ = task_id_counter_ % INT_MAX;
        py::buffer_info info = image.request();
        cv::Mat mat(info.shape[0], info.shape[1], CV_8UC3, info.ptr);
        { 
            std::lock_guard<std::mutex> lock(queue_mutex_);
            task_queue_.push({mat, task_id});
        }
        cv_new_task_.notify_one();
        // std::cout << "添加图片完成" << std::endl;
        return task_id;
    }

    /* 终止传图，目前不满 Batch Size 的也立即全部处理 */
    void synchronize() {
        /* 等待所有任务全部分发完 */
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cv_task_queue_empty_.wait(lock, [this]() { return task_queue_.empty(); });
        for (auto& worker : workers_) {
            WorkerSharedBuffer *buffer = worker->get_buffer();
            buffer->force_ready();
        }
        while (true) {
            bool all_done = true;
            for (auto& worker : workers_) {
                WorkerSharedBuffer *buffer = worker->get_buffer();
                if (!buffer->all_filling()) { all_done = false; }
            }
            if (all_done) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // printf("[dispatcher.hpp synchronize] All tasks have been dispatched and processed.\n");
    }

    /* 拿结果（非阻塞式）*/
    pyTaskResultType get_result(int task_id) {
        std::lock_guard<std::mutex> lock(result_mutex_);
        if (task_results_.find(task_id) == task_results_.end()) {
            return py::dict{};
            // return py_task_no_result;
        }  
        auto result_ptr = task_results_[task_id];  // 先保存 shared_ptr (计数+1)
        task_results_.erase(task_id);              // 从map移除，但对象不会销毁
        TaskResult& result = *result_ptr;          // 仍然安全可用
        // std::cout << "result===> " << result.value << std::endl;
        // std::cout << "result===> " << result.data << std::endl;
        return convert_to_py_task_result(result);
    }
    

private:

    void worker_task_dispatching_loop_(int index) {
        TaskWorkerWrapper *worker = workers_[index];
        WorkerSharedBuffer *buffer = worker->get_buffer();

        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_new_task_.wait(lock, [this] {
                    return !task_queue_.empty() || dispatching_threads_stopped_;
                });
                if (dispatching_threads_stopped_ && task_queue_.empty()) break;
                task = task_queue_.front();
                task_queue_.pop();
                if (task_queue_.empty()) cv_task_queue_empty_.notify_all();
            }
            int slot_id = buffer->wait_next_available_block();
            buffer->upload_task(task, slot_id);
            if (ENABLE_LOGGING)
                LOGF("Worker Wrapper Loop %d will upload task ID: %d to slot ID: %d",index, task.task_id, slot_id);
        }
    }

    void worker_result_fetching_loop_(int index) {
        TaskWorkerWrapper *worker = workers_[index];
        WorkerSharedBuffer *buffer = worker->get_buffer();
        while (true) {
            int slot_id = buffer->wait_next_done_block();
            if (slot_id == -1) break;
            bip::scoped_lock<bip::interprocess_mutex> lock(buffer->blocks[slot_id].mutex_block_data_);
            for (int i = 0; i < BATCH_SIZE; ++i) {
                if (buffer->blocks[slot_id].task_batch_[i].data_filled) {
                    TaskResult& result = buffer->blocks[slot_id].task_batch_[i].result;
                    if (ENABLE_LOGGING)
                        LOGF("Worker Loop %d received result of task ID: %d",index, buffer->blocks[slot_id].task_batch_[i].task_id);
                    buffer->blocks[slot_id].task_batch_[i].data_filled = false;
                    {
                        std::lock_guard<std::mutex> result_lock(result_mutex_);
                        if (task_results_.find(buffer->blocks[slot_id].task_batch_[i].task_id) != task_results_.end()) {
                            printf("[dispatcher.hpp worker_result_fetching_loop_] Warning: Overwriting existing result for task ID: %d\n",
                                   buffer->blocks[slot_id].task_batch_[i].task_id);
                        }
                        // TaskResult new_result;
                        // TaskResult* result_copy = new TaskResult(result); 
                        task_results_[buffer->blocks[slot_id].task_batch_[i].task_id] = std::make_shared<TaskResult>(result); /* 在这里做一次深拷贝 */
                    }
                }
            }
            buffer->blocks[slot_id].task_status = FILLING;
            buffer->cv_slot_available_.notify_all();
            if (result_fetching_threads_stopped_) break;
        }
        // printf("[dispatcher.hpp worker_result_fetching_loop_] Worker Fetcher Loop %d stoppeds.\n", index);
    }

    std::queue<Task>                    task_queue_;
    std::map<int, std::shared_ptr<TaskResult>>           task_results_;
    std::vector<TaskWorkerWrapper *>    workers_;
    std::vector<std::thread>            worker_task_dispatching_threads_;
    std::vector<std::thread>            worker_result_fetching_threads_;
    bool                                dispatching_threads_stopped_ = false;
    bool                                result_fetching_threads_stopped_ = false;
    std::mutex                          queue_mutex_;
    std::mutex                          result_mutex_;
    std::condition_variable             cv_new_task_;
    std::condition_variable             cv_task_queue_empty_;
    int                                 task_id_counter_ = 0;

};
