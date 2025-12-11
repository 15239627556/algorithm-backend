#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include "common.hpp"
#include <condition_variable>
#include "libx40Enhance.hpp"
#include <functional>
#include <chrono>
#define PI                  acos(-1)
#define INPUT_IMAGE_NUM     4
#define BLOCK_NUM           4

struct X40EnhanceTask {
        WorkerSharedBuffer* buffer_;
        int slot_id_;
        bool flag_x40Enhance_inferred = false;
        cv::Mat out_image;
        cv::Mat image;
    };

class X40EnhanceTaskWorker {
public:
    using TaskSelector = std::function<bool(const std::shared_ptr<X40EnhanceTask>&)>;
    using TaskProcessor = std::function<void(const std::shared_ptr<X40EnhanceTask>&)>;
    using Notifier = std::function<void()>;

    X40EnhanceTaskWorker(std::string model_name,
                  std::deque<std::shared_ptr<X40EnhanceTask>>& tasks,
                  std::mutex& mutex,
                  std::condition_variable& cv_wait,
                  TaskSelector should_run,
                  TaskProcessor process,
                  Notifier notify_next,
                  std::atomic<bool>& stop_flag)
        : name_(std::move(model_name)),
          task_queue_(tasks),
          mutex_(mutex),
          cv_(cv_wait),
          should_process_(should_run),
          process_task_(process),
          notify_next_(notify_next),
          stop_flag_(stop_flag)
    {}

    void operator()() {
        while (true) {
            std::shared_ptr<X40EnhanceTask> task = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&]() {
                    return stop_flag_ || std::any_of(task_queue_.begin(), task_queue_.end(), should_process_);
                });
                if (stop_flag_) break;
                for (auto& t : task_queue_) {
                    if (should_process_(t)) {
                        task = t; // 拷贝 shared_ptr，线程安全
                        break;
                    }
                }
            }

            if (task) {
                process_task_(task);
                notify_next_();
            }
        }
        std::cout << "[Worker] " << name_ << " exit.\n";
    }

private:
    std::string name_;
    std::deque<std::shared_ptr<X40EnhanceTask>>& task_queue_;
    std::mutex& mutex_;
    std::condition_variable& cv_;
    TaskSelector should_process_;
    TaskProcessor process_task_;
    Notifier notify_next_;
    std::atomic<bool>& stop_flag_;
};


class X40Enhance{

public:
    X40Enhance(int wid, int gpu_id) : worker_id(wid) {
        /*模型加载*/
        lp40xEnhance = new EnhanceOnnx(gpu_id);
        if (!lp40xEnhance) {
            std::cerr << "EnhanceOnnx 加载失败！" << std::endl;
        }

        std::cout << "模型加载结束" << std::endl;

        stop_all = false;

        /*开启子线程*/
        start_workers();
        result_thread_ = std::thread(&X40Enhance::result_thread, this);
    }
    ~X40Enhance(){
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            stop_all = true;
        }

        cv_infer.notify_all();
        cv_result.notify_all();

        /* 回收封装的上传/推理线程 */
        for (auto& t : all_threads_) {
            if (t.joinable()) t.join();
        }

        /* 回收汇总线程 */
        if (result_thread_.joinable()) result_thread_.join();

    }
    /*添加任务到队列*/
    void add_x40_enhance_task(cv::Mat img, WorkerSharedBuffer* buffer, int slot_id) {
        auto task = std::make_shared<X40EnhanceTask>();
        task->image = img + 0;
        task->buffer_ = buffer;
        task->slot_id_ = slot_id;
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            task_queue.emplace_back(task);
        }
        cv_infer.notify_all();
    }

private:
    std::vector<std::thread> all_threads_;  // 存储所有X40EnhanceTaskWorker线程
    std::deque<std::shared_ptr<X40EnhanceTask>> task_queue;
    std::mutex task_mutex;
    std::condition_variable cv_infer;
    std::condition_variable cv_result;

    std::atomic<bool> stop_all = false;
    int worker_id;

    std::thread result_thread_;
    std::thread test_thread_;

    EnhanceOnnx *lp40xEnhance;

    
     /*
        ******************************数据整合**********************************
    */
    void data_merge(std::shared_ptr<X40EnhanceTask> &task)
    {

        {
            WorkerDataBlock &block = task->buffer_->blocks[task->slot_id_];
            TaskDataBlock &result_ = block.task_batch_[0];
            TaskResult &result = result_.result;
            //滤镜结果赋值
            if (task->out_image.isContinuous()) {
                std::memcpy(result.data, task->out_image.data, 2448*2048*3);
            } else {
                const size_t row_bytes = static_cast<size_t>(2448) * 3;
                for (int r = 0; r < 2048; ++r) {
                    std::memcpy(result.data + r * row_bytes, task->out_image.ptr(r), row_bytes);
                }
            }

            // cv::imwrite("out.jpg", task->out_image);
                
            block.task_status = DONE;  
            task->buffer_->cv_result_ready_.notify_all();
            for (int i = 0; i < BATCH_SIZE; ++i) {
                TaskDataBlock &task = block.task_batch_[i];
                if (!task.data_filled) continue;
                LOGF("Worker Process %d completed task %d", worker_id, task.task_id);
            }
        }
    }


    void start_workers() {
        all_threads_.emplace_back([this]() {
            X40EnhanceTaskWorker worker(
                "Infer-X40Enhance",
                task_queue, task_mutex, cv_infer,
                [](const std::shared_ptr<X40EnhanceTask>& t) {
                    return !t->flag_x40Enhance_inferred;
                },
                [this](const std::shared_ptr<X40EnhanceTask>& t) {
                    std::cout << "1111111111" << std::endl;
                    t->flag_x40Enhance_inferred = lp40xEnhance->infer(t->image, t->out_image);
                },
                [&]() { cv_result.notify_one(); },
                stop_all
            );
            worker();
        });

    }

    
    void result_thread() {
        while (true) {
            std::shared_ptr<X40EnhanceTask> task_to_collect;
            {
                std::unique_lock<std::mutex> lock(task_mutex);
                /*找到一个完成所有模型计算的任务*/
                cv_result.wait(lock, [&] {
                    if (stop_all) return true;
                    for (auto &t : task_queue) {

                        if (t->flag_x40Enhance_inferred) {
                            return true;
                        }
                    }
                    return false;

                });
                if (stop_all) break;

                for (auto it = task_queue.begin(); it != task_queue.end(); ++it) {
                    if ((*it)->flag_x40Enhance_inferred) {
                        task_to_collect = *it;
                        task_queue.erase(it);
                        break;
                    }
                }
            }
            data_merge(task_to_collect);
        }
    }
};