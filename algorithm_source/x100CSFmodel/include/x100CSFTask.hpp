#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include "common.hpp"
#include <condition_variable>
#include "libX100CSFCellLoc.hpp"
#include "lib100XCSFClassify.hpp"
#include <functional>
#include <chrono>
#define INPUT_IMAGE_NUM     1
#define INPUT_CELL_NUM      32



struct X100CSFTask {
    WorkerSharedBuffer* buffer_;
    int slot_id_;
    std::vector<cv::Mat> images;
    int image_actual_num = INPUT_IMAGE_NUM;
    bool flag_x100_csf_locate_inferred = false;
    bool flag_x100_csf_classify_inferred = false;
    // std::vector<itemX100CSFLocateInfo> result_x100_csf_locate;
    std::vector<itmCellRcgz_x100CSF_image> result_x100_csf_locate;
    std::vector<std::vector<itmCellRcgz_x100CSF_image>> result_x100_csf_classify;

};


class X100CSFMain{

public:
    X100CSFMain(int wid, int gpu_id) : worker_id(wid) {
        /*模型加载*/
        lpLocation100xCSF = new CSFCellLocateOnnx(gpu_id);
        if (!lpLocation100xCSF) {
            std::cerr << "CSFCellLocateOnnx 加载失败！" << std::endl;
        }

	    lpClassify100xCSF = new libOnnxMNISTCSF(gpu_id);
        if (!lpClassify100xCSF) {
            std::cerr << "libOnnxMNISTCSF 加载失败！" << std::endl;
        }

        std::cout << "模型加载结束" << std::endl;

        stop_all = false;

        /*开启子线程*/
        result_csf_thread_ = std::thread(&X100CSFMain::result_csf_thread, this);
        infer_csf_locate_thread_ = std::thread(&X100CSFMain::infer_csf_locate_thread, this);
        infer_csf_classify_thread_ = std::thread(&X100CSFMain::infer_csf_classify_thread, this);
    }
    ~X100CSFMain(){
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            stop_all = true;
        }

        cv_infer_csf_locate.notify_all();
        cv_infer_csf_classify.notify_all();
        cv_csf_result.notify_all();

        /*回收计算线程*/
        if (infer_csf_locate_thread_.joinable()) infer_csf_locate_thread_.join();
        if (infer_csf_classify_thread_.joinable()) infer_csf_classify_thread_.join();
        /* 回收汇总线程 */
        if (result_csf_thread_.joinable()) result_csf_thread_.join();


    }
    /*添加任务到队列*/
    void add_csf_task(std::vector<cv::Mat>& imgs, WorkerSharedBuffer* buffer, int slot_id) {
        const int image_actual_num_local = static_cast<int>(imgs.size());
        std::vector<cv::Mat> padded = imgs;      // 避免在原imgs上反复push_back
        for (int i = image_actual_num_local; i < INPUT_IMAGE_NUM; ++i) {
            padded.push_back(padded[0]);         // padding
        }
        auto task = std::make_shared<X100CSFTask>();
        task->images = padded;
        task->buffer_ = buffer;
        task->slot_id_ = slot_id;
        task->image_actual_num = image_actual_num_local; // 建议放到task里，不用全局
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            task_queue.emplace_back(task);
        }
        
        cv_infer_csf_locate.notify_one();
        //std::this_thread::yield();
    }

private:
    std::deque<std::shared_ptr<X100CSFTask>> task_queue;
    std::mutex task_mutex;
    std::condition_variable cv_infer_csf_locate;
    std::condition_variable cv_infer_csf_classify;
    std::condition_variable cv_csf_result;

    std::atomic<bool> stop_all = false;
    int worker_id;

    std::thread result_csf_thread_;
    std::thread infer_csf_locate_thread_;
    std::thread infer_csf_classify_thread_;

    CSFCellLocateOnnx* lpLocation100xCSF;
    libOnnxMNISTCSF* lpClassify100xCSF;
    
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
    void data_merge_csf(std::shared_ptr<X100CSFTask> &task)
    {
        {
            WorkerDataBlock &block = task->buffer_->blocks[task->slot_id_];
            size_t batchsize = task->image_actual_num < task->result_x100_csf_classify.size() ? task->image_actual_num : task->result_x100_csf_classify.size();
            
            TaskDataBlock &result_ = block.task_batch_[0];
            for(size_t i = 0; i < task->result_x100_csf_locate.size(); i++)
            {
                result_.result.imageResultInfos.cellRectsSize = task->result_x100_csf_locate.size();
                result_.result.imageResultInfos.cellRects[i].x = task->result_x100_csf_locate[i].cellRect.x;
                result_.result.imageResultInfos.cellRects[i].y = task->result_x100_csf_locate[i].cellRect.y;
                result_.result.imageResultInfos.cellRects[i].w = task->result_x100_csf_locate[i].cellRect.width;
                result_.result.imageResultInfos.cellRects[i].h = task->result_x100_csf_locate[i].cellRect.height;

                

                result_.result.imageResultInfos.cellClassifyResult[i].top1 = task->result_x100_csf_locate[i].top5list[0].m_type;
                result_.result.imageResultInfos.cellClassifyResult[i].top2 = task->result_x100_csf_locate[i].top5list[1].m_type;
                result_.result.imageResultInfos.cellClassifyResult[i].top3 = task->result_x100_csf_locate[i].top5list[2].m_type;
                result_.result.imageResultInfos.cellClassifyResult[i].top4 = task->result_x100_csf_locate[i].top5list[3].m_type;
                result_.result.imageResultInfos.cellClassifyResult[i].top5 = task->result_x100_csf_locate[i].top5list[4].m_type;

                result_.result.imageResultInfos.cellClassifyResult[i].ratio1 = task->result_x100_csf_locate[i].top5list[0].m_pcnt;
                result_.result.imageResultInfos.cellClassifyResult[i].ratio2 = task->result_x100_csf_locate[i].top5list[1].m_pcnt;
                result_.result.imageResultInfos.cellClassifyResult[i].ratio3 = task->result_x100_csf_locate[i].top5list[2].m_pcnt;
                result_.result.imageResultInfos.cellClassifyResult[i].ratio4 = task->result_x100_csf_locate[i].top5list[3].m_pcnt;
                result_.result.imageResultInfos.cellClassifyResult[i].ratio5 = task->result_x100_csf_locate[i].top5list[4].m_pcnt;

            }
            // for(size_t b = 0; b < batchsize; b++)
            // {
            //     TaskDataBlock &result_ = block.task_batch_[b];
               
            //     //细胞信息赋值
            //     result_.result.imageResultInfos.cellRectsSize = task->result_x100_csf_classify[b].size();
            //     for(size_t i = 0; i < task->result_x100_csf_classify[b].size(); i++)
            //     {
            //         result_.result.imageResultInfos.cellRects[i].x = task->result_x100_csf_classify[b][i].cellRect.x;
            //         result_.result.imageResultInfos.cellRects[i].y = task->result_x100_csf_classify[b][i].cellRect.y;
            //         result_.result.imageResultInfos.cellRects[i].w = task->result_x100_csf_classify[b][i].cellRect.width;
            //         result_.result.imageResultInfos.cellRects[i].h = task->result_x100_csf_classify[b][i].cellRect.height;

            //         result_.result.imageResultInfos.cellClassifyResult[i].top1 = task->result_x100_csf_classify[b][i].top5list[0].m_type;
            //         result_.result.imageResultInfos.cellClassifyResult[i].top2 = task->result_x100_csf_classify[b][i].top5list[1].m_type;
            //         result_.result.imageResultInfos.cellClassifyResult[i].top3 = task->result_x100_csf_classify[b][i].top5list[2].m_type;
            //         result_.result.imageResultInfos.cellClassifyResult[i].top4 = task->result_x100_csf_classify[b][i].top5list[3].m_type;
            //         result_.result.imageResultInfos.cellClassifyResult[i].top5 = task->result_x100_csf_classify[b][i].top5list[4].m_type;

            //         result_.result.imageResultInfos.cellClassifyResult[i].ratio1 = task->result_x100_csf_classify[b][i].top5list[0].m_pcnt;
            //         result_.result.imageResultInfos.cellClassifyResult[i].ratio2 = task->result_x100_csf_classify[b][i].top5list[1].m_pcnt;
            //         result_.result.imageResultInfos.cellClassifyResult[i].ratio3 = task->result_x100_csf_classify[b][i].top5list[2].m_pcnt;
            //         result_.result.imageResultInfos.cellClassifyResult[i].ratio4 = task->result_x100_csf_classify[b][i].top5list[3].m_pcnt;
            //         result_.result.imageResultInfos.cellClassifyResult[i].ratio5 = task->result_x100_csf_classify[b][i].top5list[4].m_pcnt;
            //     }
                
            // }   
            block.task_status = DONE;  
            task->buffer_->cv_result_ready_.notify_all();
            for (int i = 0; i < BATCH_SIZE; ++i) {
                TaskDataBlock &task = block.task_batch_[i];
                if (!task.data_filled) continue;
                LOGF("Worker Process %d completed task %d", worker_id, task.task_id);
            }
        }
    }

    void infer_csf_locate_thread() {
        while (true) {
            std::shared_ptr<X100CSFTask> task_to_collect;
            {
                std::unique_lock<std::mutex> lock(task_mutex);
                cv_infer_csf_locate.wait(lock, [&] {
                    if (stop_all) return true;
                    for (auto &t : task_queue) {
                        if (!t->flag_x100_csf_locate_inferred) {
                            return true;
                        }
                    }
                    return false;
                });
                if (stop_all) break;

                for (auto it = task_queue.begin(); it != task_queue.end(); ++it) {
                    if (!(*it)->flag_x100_csf_locate_inferred) {
                        task_to_collect = *it;
                        break;
                    }
                }
            }
            std::vector<itemX100CSFLocateInfo> x100_csf_locate_out;
            task_to_collect->flag_x100_csf_locate_inferred = lpLocation100xCSF->infer(task_to_collect->images, x100_csf_locate_out);
            //所有采样图上的所有细胞
            for(size_t i = 0; i < x100_csf_locate_out.size(); i++)
            {
                for(size_t cell = 0; cell < x100_csf_locate_out[i].boxes.rows; cell++)
                {
                    itmCellRcgz_x100CSF_image cellinfos;
                    const float* box = x100_csf_locate_out[i].boxes.ptr<float>(cell);
                    cv::Rect cell_rect = cv::Rect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
                    cellinfos.cellRect = std::move(cell_rect);
                    cellinfos.picIndex = i; //细胞所在采样图下标
                    cellinfos.cellisempty = false;
                    cellinfos.top5list = {};
                    task_to_collect->result_x100_csf_locate.push_back(cellinfos); 
                }
            }
            cv_infer_csf_classify.notify_one();
        }
    }

    void infer_csf_classify_thread() {
        while (true) {
            std::shared_ptr<X100CSFTask> task_to_collect;
            {
                std::unique_lock<std::mutex> lock(task_mutex);
                cv_infer_csf_classify.wait(lock, [&] {
                    if (stop_all) return true;
                    for (auto &t : task_queue) {
                        if (!t->flag_x100_csf_classify_inferred && t->flag_x100_csf_locate_inferred) {
                            return true;
                        }
                    }
                    return false;
                });
                if (stop_all) break;

                for (auto it = task_queue.begin(); it != task_queue.end(); ++it) {
                    if (!(*it)->flag_x100_csf_classify_inferred && (*it)->flag_x100_csf_locate_inferred) {
                        task_to_collect = *it;
                        break;
                    }
                }
            }
            if(task_to_collect->result_x100_csf_locate.size() <= 0)
            {
                task_to_collect->flag_x100_csf_classify_inferred = true;
            }
            else
            {
                std::vector<cellMatAndIndex> cellMatList;
                //循环细胞
                for(size_t i = 0; i < task_to_collect->result_x100_csf_locate.size(); i++)
                {
                    int picIndex = task_to_collect->result_x100_csf_locate[i].picIndex;
                    
                    //用来补齐采样图列表的图片上细胞不需要分类
                    if(picIndex >= task_to_collect->image_actual_num)
                    {
                        task_to_collect->result_x100_csf_locate[i].cellisempty = true;
                        std::cout << "picIndex == " << picIndex << "task_to_collect->image_actual_num == " << task_to_collect->image_actual_num << std::endl;
                        continue;
                    }
                    cellMatAndIndex cellmatAndindex;
                    std::vector<itmCellRcgz_x100CSF_image> outlist;
                    //补正方形
                    cv::Rect rect_new = makeUpSquare(task_to_collect->result_x100_csf_locate[i].cellRect, task_to_collect->images[picIndex]);
                    cellmatAndindex.cellMat = (task_to_collect->images[picIndex])(rect_new);
                    cellmatAndindex.cellIndex = i;
                    
                    if(cellmatAndindex.cellMat.empty())
                    {
                        task_to_collect->result_x100_csf_locate[i].cellisempty = true;
                        std::cout << "图片数据空 rect_new ----> " << rect_new << std::endl;
                        continue; 
                    }
                    //如果细胞列表中细胞数量不够INPUT_CELL_NUM，继续添加
                    if(cellMatList.size() < INPUT_CELL_NUM)
                    {
                        cellMatList.push_back(cellmatAndindex);
                        continue;
                    }
                    //如果细胞列表中细胞数量==INPUT_CELL_NUM，进入模型计算
                    std::vector<std::vector<itmCellRcgz_x100CSF>> out2; //INPUT_CELL_NUM张图的top5+rate5
                    lpClassify100xCSF->infer(cellMatList, out2);
                    for(int j = 0; j < out2.size(); j++)
                    {
                        int now_index = cellMatList[j].cellIndex;
                        for(int k = 0; k < 5; k++)
                        {
                            task_to_collect->result_x100_csf_locate[now_index].top5list.push_back(out2[j][k]);
                        }
                    }
                    cellMatList.clear();   
                    cellMatList.push_back(cellmatAndindex);                
                }
                //循环结束，有剩余细胞且数量小于INPUT_CELL_NUM，补齐计算
                if(cellMatList.size() > 0 && cellMatList.size() < INPUT_CELL_NUM)
                {
                    //补全
                    int actual_cell_num = cellMatList.size();
                    for(int j = actual_cell_num; j < INPUT_CELL_NUM; j++)
                    {
                        cellMatList.push_back(cellMatList[0]);
                    }
                    std::vector<std::vector<itmCellRcgz_x100CSF>> out3; //INPUT_CELL_NUM张图的top5+rate5
                    lpClassify100xCSF->infer(cellMatList, out3);
                    for(int j = 0; j < actual_cell_num; j++)
                    {
                        int now_index = cellMatList[j].cellIndex;
                        for(int k = 0; k < 5; k++)
                        {
                            task_to_collect->result_x100_csf_locate[now_index].top5list.push_back(out3[j][k]);
                        }
                    }
                    cellMatList.clear();   
                }
                //删除未进行分类（细胞数据空或者补齐的采样图上的细胞）的细胞信息
                task_to_collect->result_x100_csf_locate.erase(
                    std::remove_if(
                        task_to_collect->result_x100_csf_locate.begin(),
                        task_to_collect->result_x100_csf_locate.end(),
                        [](const itmCellRcgz_x100CSF_image &img) {
                            return img.cellisempty;  // 删除 cellisempty == true 的元素
                        }),
                    task_to_collect->result_x100_csf_locate.end());

 
                // // 预分配层级容器
                // task_to_collect->result_x100_csf_classify.resize(INPUT_IMAGE_NUM);
                // std::cout << "task_to_collect->result_x100_csf_classify == " << task_to_collect->result_x100_csf_classify.size() << std::endl;
                // // 按 picIndex 分类归层
                // for (int j = 0; j < task_to_collect->result_x100_csf_locate.size(); j++) {
                //     int idx = task_to_collect->result_x100_csf_locate[j].picIndex;
                //     if (idx < 0 || idx >= INPUT_IMAGE_NUM) continue; // 防御性检查
                //     task_to_collect->result_x100_csf_classify[idx].push_back(task_to_collect->result_x100_csf_locate[j]);
                // }

                task_to_collect->flag_x100_csf_classify_inferred = true;
            }
            cv_csf_result.notify_one();
        }
    }
    
    void result_csf_thread() {
        while (true) {
            std::shared_ptr<X100CSFTask> task_to_collect;
            {
                std::unique_lock<std::mutex> lock(task_mutex);
                /*找到一个完成所有模型计算的任务*/
                cv_csf_result.wait(lock, [&] {
                    if (stop_all) return true;
                    for (auto &t : task_queue) {
                        if (t->flag_x100_csf_classify_inferred && t->flag_x100_csf_locate_inferred) {
                            return true;
                        }
                    }
                    return false;

                });
                if (stop_all) break;

                for (auto it = task_queue.begin(); it != task_queue.end(); ++it) {
                    if ((*it)->flag_x100_csf_classify_inferred && (*it)->flag_x100_csf_locate_inferred) {
                        task_to_collect = *it;
                        task_queue.erase(it);
                        break;
                    }
                }
            }
            data_merge_csf(task_to_collect);
        }
    }
};