#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include "common.hpp"
#include <condition_variable>
#include "libOnnxMNIST.hpp"
#include "libX100BigCellClassify.hpp"
#include "libX100BigCellLoc.hpp"
#include "libX100CellLoc.hpp"
#include <functional>
#include <chrono>
#define INPUT_IMAGE_NUM     1

struct X100Task {
    TaskTypes tasktype;
    WorkerSharedBuffer* buffer_;
    int slot_id_;
    cv::Mat images;
    int image_actual_num = INPUT_IMAGE_NUM;
    bool flag_x100_locate_inferred = false;
    bool flag_x100_classify_inferred = false;
    std::vector<cv::Rect> result_x100_locate;
    std::vector<std::vector<itmCellRcgz_x100>> result_x100_classify;
};


class X100Main{

public:
    X100Main(int wid, int gpu_id) : worker_id(wid) {
        /*模型加载*/
        lpLocation100xHave = new X100HaveLocateOnnx(gpu_id);
        if (!lpLocation100xHave) {
            std::cerr << "X100HaveLocateOnnx 加载失败！" << std::endl;
        }

	    lpClassify100xHave = new X100HaveClassifyOnnx(gpu_id);
        if (!lpClassify100xHave) {
            std::cerr << "X100HaveClassifyOnnx 加载失败！" << std::endl;
        }

        lplocation100XBig = new X100BigLocateOnnx(gpu_id);
        if (!lplocation100XBig) {
            std::cerr << "X100BigLocateOnnx 加载失败！" << std::endl;
        }

        lpClassify100XBig = new X100BigClassifyOnnx(gpu_id);
        if (!lpClassify100XBig) {
            std::cerr << "X100BigClassifyOnnx 加载失败！" << std::endl;
        }


        std::cout << "模型加载结束" << std::endl;

        stop_all = false;

        /*开启子线程*/
        result_thread_ = std::thread(&X100Main::result_thread, this);
        infer_locate_thread_ = std::thread(&X100Main::infer_locate_thread, this);
        infer_classify_thread_ = std::thread(&X100Main::infer_classify_thread, this);
    }
    ~X100Main(){
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            stop_all = true;
        }

        cv_infer_locate.notify_all();
        cv_infer_classify.notify_all();
        cv_result.notify_all();

        /*回收计算线程*/
        if (infer_locate_thread_.joinable()) infer_locate_thread_.join();
        if (infer_classify_thread_.joinable()) infer_classify_thread_.join();
        /* 回收汇总线程 */
        if (result_thread_.joinable()) result_thread_.join();


    }
    /*添加任务到队列*/
    void add_x40_task(std::vector<cv::Mat>& imgs, WorkerSharedBuffer* buffer, int slot_id, TaskTypes tasktype) {
        const int image_actual_num_local = static_cast<int>(imgs.size());
        auto task = std::make_shared<X100Task>();
        task->images = imgs[0];
        task->buffer_ = buffer;
        task->slot_id_ = slot_id;
        task->tasktype = tasktype;
        task->image_actual_num = image_actual_num_local; // 建议放到task里，不用全局
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            task_queue.emplace_back(task);
        }
        cv_infer_locate.notify_one();
        std::this_thread::yield();
    }

private:
    std::deque<std::shared_ptr<X100Task>> task_queue;
    std::mutex task_mutex;
    std::condition_variable cv_infer_locate;
    std::condition_variable cv_infer_classify;
    std::condition_variable cv_result;

    std::atomic<bool> stop_all = false;
    int worker_id;

    std::thread result_thread_;
    std::thread infer_locate_thread_;
    std::thread infer_classify_thread_;

    X100HaveLocateOnnx* lpLocation100xHave;
    X100HaveClassifyOnnx* lpClassify100xHave;
    X100BigLocateOnnx* lplocation100XBig;
    X100BigClassifyOnnx* lpClassify100XBig;
    
    /*图片处理*/
    cv::Rect makeUpSquare(cv::Rect re, cv::Mat flame)
    {
        int temp, temp1, temp2, temp3, temp4;
        if (re.height > re.width)
        {
            temp = 0;
            temp1 = re.height - re.width;
            temp2 = flame.cols;
            temp3 = re.width;
            temp4 = re.x;
            re.width = re.height;
        }
        else {
            temp = 1;
            temp1 = re.width - re.height;
            temp2 = flame.rows;
            temp3 = re.height;
            temp4 = re.y;
            re.height = re.width;
        }
        if (temp1 % 2 == 0)
        {
            if (temp4 - temp1 / 2 >= 0 && temp4 + temp3 + temp1 / 2 <= temp2)
            {
                temp4 = temp4 - temp1 / 2;
            }
            else if (temp4 - temp1 / 2 < 0 && temp4 + temp3 + temp1 / 2 <= temp2)
            {
                temp4 = 0;

            }
            else if (temp4 - temp1 / 2 >= 0 && temp4 + temp3 + temp1 / 2 > temp2)
            {
                temp4 = temp2 - re.width;
            }
        }
        else
        {
            if (temp4 - (temp1 + 1) / 2 >= 0 && temp4 + temp3 + (temp1 - 1) / 2 <= temp2)
            {
                temp4 = temp4 - (temp1 + 1) / 2;
            }
            else if (temp4 - (temp1 + 1) / 2 < 0 && temp4 + temp3 + (temp1 - 1) / 2 <= temp2)
            {
                temp4 = 0;

            }
            else if (temp4 - (temp1 + 1) / 2 >= 0 && temp4 + temp3 + (temp1 - 1) / 2 > temp2)
            {
                temp4 = temp2 - re.width;
            }
        }
        if (temp == 0)
        {
            re.x = temp4;
        }
        else
        {
            re.y = temp4;
        }
        cv::Rect rect_(0, 0, flame.cols, flame.rows);
        re = re & rect_;
        return re;
    }

    
     /*
        ******************************数据整合**********************************
    */
    void data_merge_x100(std::shared_ptr<X100Task> &task)
    {
        {
            WorkerDataBlock &block = task->buffer_->blocks[task->slot_id_];
            for(int b = 0; b < 1; b++)
            {
                TaskDataBlock &result_ = block.task_batch_[b];
               
                //细胞信息赋值
                result_.result.imageResultInfos.cellRectsSize = task->result_x100_locate.size();
                for(size_t i = 0; i < task->result_x100_locate.size(); i++)
                {
                    result_.result.imageResultInfos.cellRects[i].x = task->result_x100_locate[i].x;
                    result_.result.imageResultInfos.cellRects[i].y = task->result_x100_locate[i].y;
                    result_.result.imageResultInfos.cellRects[i].w = task->result_x100_locate[i].width;
                    result_.result.imageResultInfos.cellRects[i].h = task->result_x100_locate[i].height;

                    result_.result.imageResultInfos.cellClassifyResult[i].top1 = task->result_x100_classify[i][0].m_type;
                    result_.result.imageResultInfos.cellClassifyResult[i].top2 = task->result_x100_classify[i][1].m_type;
                    result_.result.imageResultInfos.cellClassifyResult[i].top3 = task->result_x100_classify[i][2].m_type;
                    result_.result.imageResultInfos.cellClassifyResult[i].top4 = task->result_x100_classify[i][3].m_type;
                    result_.result.imageResultInfos.cellClassifyResult[i].top5 = task->result_x100_classify[i][4].m_type;

                    result_.result.imageResultInfos.cellClassifyResult[i].ratio1 = task->result_x100_classify[i][0].m_pcnt;
                    result_.result.imageResultInfos.cellClassifyResult[i].ratio2 = task->result_x100_classify[i][1].m_pcnt;
                    result_.result.imageResultInfos.cellClassifyResult[i].ratio3 = task->result_x100_classify[i][2].m_pcnt;
                    result_.result.imageResultInfos.cellClassifyResult[i].ratio4 = task->result_x100_classify[i][3].m_pcnt;
                    result_.result.imageResultInfos.cellClassifyResult[i].ratio5 = task->result_x100_classify[i][4].m_pcnt;
                }
                
            }   
            block.task_status = DONE;  
            task->buffer_->cv_result_ready_.notify_all();
            for (int i = 0; i < BATCH_SIZE; ++i) {
                TaskDataBlock &task = block.task_batch_[i];
                if (!task.data_filled) continue;
                LOGF("Worker Process %d completed task %d", worker_id, task.task_id);
            }
        }
    }

    void infer_locate_thread() {
        while (true) {
            std::shared_ptr<X100Task> task_to_collect;
            {
                std::unique_lock<std::mutex> lock(task_mutex);
                cv_infer_locate.wait(lock, [&] {
                    if (stop_all) return true;
                    for (auto &t : task_queue) {
                        if (!t->flag_x100_locate_inferred) {
                            return true;
                        }
                    }
                    return false;
                });
                if (stop_all) break;

                for (auto it = task_queue.begin(); it != task_queue.end(); ++it) {
                    if (!(*it)->flag_x100_locate_inferred) {
                        task_to_collect = *it;
                        break;
                    }
                }
            }
            
            if(task_to_collect->tasktype == X100HAVECELL)
            {
                std::cout << "X100HAVECELL  locate" << std::endl;
                task_to_collect->flag_x100_locate_inferred = lpLocation100xHave->infer(task_to_collect->images, task_to_collect->result_x100_locate);
            }
            else{
                std::cout << "X100HAVECELL big locate" << std::endl;
                task_to_collect->flag_x100_locate_inferred = lplocation100XBig->infer(task_to_collect->images, task_to_collect->result_x100_locate);
            }
            std::cout << "task_to_collect->result_x100_locate " << task_to_collect->result_x100_locate.size() << std::endl;
            cv_infer_classify.notify_one();
        }
    }

    void infer_classify_thread() {
        while (true) {
            std::shared_ptr<X100Task> task_to_collect;
            {
                std::unique_lock<std::mutex> lock(task_mutex);
                cv_infer_classify.wait(lock, [&] {
                    if (stop_all) return true;
                    for (auto &t : task_queue) {
                        if (!t->flag_x100_classify_inferred && t->flag_x100_locate_inferred) {
                            return true;
                        }
                    }
                    return false;
                });
                if (stop_all) break;

                for (auto it = task_queue.begin(); it != task_queue.end(); ++it) {
                    if (!(*it)->flag_x100_classify_inferred && (*it)->flag_x100_locate_inferred) {
                        task_to_collect = *it;
                        break;
                    }
                }
            }
            if(task_to_collect->result_x100_locate.size() <= 0)
            {
                task_to_collect->flag_x100_classify_inferred = true;
            }
            else
            {
                cv::Mat src = task_to_collect->images + 0;
                std::vector<itmCellRcgz_x100> out;
                for(size_t i = 0; i < task_to_collect->result_x100_locate.size(); i++)
                {
                    cv::Rect rect_new = makeUpSquare(task_to_collect->result_x100_locate[i], src);
                    cv::Mat cell_mat = src(rect_new);
                    if(cell_mat.empty())
                    {
                        std::cout << "图片数据空 rect_new ----> " << rect_new << std::endl;
                        continue; 
                    }
                    // cv::imwrite("cell.jpg", cell_mat);
                    if(task_to_collect->tasktype == X100HAVECELL)
                    {
                        std::cout << "X100HAVECELL  classify" << std::endl;
                        lpClassify100xHave->infer(cell_mat, out);
                        out.resize(5); // 只保留前 5 个元素
                        task_to_collect->result_x100_classify.push_back(out);
                    }
                    else{
                        std::cout << "X100HAVECELL big classify" << std::endl;
                        lpClassify100XBig->infer(cell_mat, out);
                        out.resize(5); // 只保留前 5 个元素
                        task_to_collect->result_x100_classify.push_back(out);
                    }
                }
                task_to_collect->flag_x100_classify_inferred = true;
            }
            cv_result.notify_one();
        }
    }
    
    void result_thread() {
        while (true) {
            std::shared_ptr<X100Task> task_to_collect;
            {
                std::unique_lock<std::mutex> lock(task_mutex);
                /*找到一个完成所有模型计算的任务*/
                cv_result.wait(lock, [&] {
                    if (stop_all) return true;
                    for (auto &t : task_queue) {
                        if (t->flag_x100_classify_inferred && t->flag_x100_locate_inferred) {
                            return true;
                        }
                    }
                    return false;

                });
                if (stop_all) break;

                for (auto it = task_queue.begin(); it != task_queue.end(); ++it) {
                    if ((*it)->flag_x100_classify_inferred && (*it)->flag_x100_locate_inferred) {
                        task_to_collect = *it;
                        task_queue.erase(it);
                        break;
                    }
                }
            }
            data_merge_x100(task_to_collect);
        }
    }
};