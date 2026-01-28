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
#define INPUT_IMAGE_NUM     4

// 并发上限与分片粒度（按实际吞吐调）
static const int kMaxClsWorkers = 4;     // 分类线程最大数量
static const int kCellsPerShard = 32;    // 每 ~24 个细胞切 1 份分片


struct X100CSFTask {
    WorkerSharedBuffer* buffer_;
    int slot_id_;
    std::vector<cv::Mat> images;
    int image_actual_num = INPUT_IMAGE_NUM;
    bool flag_x100_csf_locate_inferred = false;
    bool flag_x100_csf_classify_inferred = false;
    std::vector<itemX100CSFLocateInfo> result_x100_csf_locate;
    std::vector<std::vector<itmCellRcgz_x100CSF_image>> result_x100_csf_classify;

    // 分类进行中与计数
    std::atomic<bool> csf_cls_in_progress{false};
    std::atomic<int>  csf_total_cells{0};
    std::atomic<int>  csf_remaining_cells{0};
};


class X100CSFMain{

public:
    X100CSFMain(int wid, int gpu_id) : worker_id(wid), gpu_id_(gpu_id) {
        /*模型加载*/
        lpLocation100xCSF = new CSFCellLocateOnnx(gpu_id);
        if (!lpLocation100xCSF) {
            std::cerr << "CSFCellLocateOnnx 加载失败！" << std::endl;
        }

	    // lpClassify100xCSF = new libOnnxMNISTCSF(gpu_id);
        // if (!lpClassify100xCSF) {
        //     std::cerr << "libOnnxMNISTCSF 加载失败！" << std::endl;
        // }

        std::cout << "模型加载结束" << std::endl;

        stop_all = false;

        /*开启子线程*/
        result_csf_thread_ = std::thread(&X100CSFMain::result_csf_thread, this);
        infer_csf_locate_thread_ = std::thread(&X100CSFMain::infer_csf_locate_thread, this);
        // infer_csf_classify_thread_ = std::thread(&X100CSFMain::infer_csf_classify_thread, this);

        // CHANGE: 不再只开一条 classify 线程；改为构建线程池
        start_csf_classify_workers(kMaxClsWorkers);
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
        // if (infer_csf_classify_thread_.joinable()) infer_csf_classify_thread_.join();
        // NEW: 停止分类线程池
        stop_csf_classify_workers();
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
    // libOnnxMNISTCSF* lpClassify100xCSF;
    int gpu_id_ = 0;

    // NEW: 分片任务
    struct CSFCellRef { int img_idx; int cell_idx; };
    struct CSFCellShardJob {
        std::shared_ptr<X100CSFTask> task;
        std::vector<CSFCellRef> cells;
    };
    std::mutex csf_job_mtx;
    std::condition_variable cv_csf_jobs;
    std::deque<CSFCellShardJob> csf_job_queue;

    // NEW: 显式实例池与线程池
    std::vector<std::unique_ptr<libOnnxMNISTCSF>> cls_instances_;
    std::vector<std::thread> csf_cls_threads;
    
    //工具函数：切分并发度、Rect 安全
    int calc_concurrency(int total_cells) {
        if (total_cells <= 0) return 1;
        int k = (total_cells + kCellsPerShard - 1) / kCellsPerShard;
        if (k < 1) k = 1;
        if (k > kMaxClsWorkers) k = kMaxClsWorkers;
        return k;
    }

    static inline cv::Rect safeRect(const cv::Rect& r, const cv::Mat& img) {
        cv::Rect bound(0,0, img.cols, img.rows);
        return (r & bound);
    }

    
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
            for(size_t b = 0; b < batchsize; b++)
            {
                TaskDataBlock &result_ = block.task_batch_[b];
               
                //细胞信息赋值
                result_.result.imageResultInfos.cellRectsSize = task->result_x100_csf_classify[b].size();
                for(size_t i = 0; i < task->result_x100_csf_classify[b].size(); i++)
                {
                    result_.result.imageResultInfos.cellRects[i].x = task->result_x100_csf_classify[b][i].cellRect.x;
                    result_.result.imageResultInfos.cellRects[i].y = task->result_x100_csf_classify[b][i].cellRect.y;
                    result_.result.imageResultInfos.cellRects[i].w = task->result_x100_csf_classify[b][i].cellRect.width;
                    result_.result.imageResultInfos.cellRects[i].h = task->result_x100_csf_classify[b][i].cellRect.height;

                    result_.result.imageResultInfos.cellClassifyResult[i].top1 = task->result_x100_csf_classify[b][i].top5list[0].m_type;
                    result_.result.imageResultInfos.cellClassifyResult[i].top2 = task->result_x100_csf_classify[b][i].top5list[1].m_type;
                    result_.result.imageResultInfos.cellClassifyResult[i].top3 = task->result_x100_csf_classify[b][i].top5list[2].m_type;
                    result_.result.imageResultInfos.cellClassifyResult[i].top4 = task->result_x100_csf_classify[b][i].top5list[3].m_type;
                    result_.result.imageResultInfos.cellClassifyResult[i].top5 = task->result_x100_csf_classify[b][i].top5list[4].m_type;

                    result_.result.imageResultInfos.cellClassifyResult[i].ratio1 = task->result_x100_csf_classify[b][i].top5list[0].m_pcnt;
                    result_.result.imageResultInfos.cellClassifyResult[i].ratio2 = task->result_x100_csf_classify[b][i].top5list[1].m_pcnt;
                    result_.result.imageResultInfos.cellClassifyResult[i].ratio3 = task->result_x100_csf_classify[b][i].top5list[2].m_pcnt;
                    result_.result.imageResultInfos.cellClassifyResult[i].ratio4 = task->result_x100_csf_classify[b][i].top5list[3].m_pcnt;
                    result_.result.imageResultInfos.cellClassifyResult[i].ratio5 = task->result_x100_csf_classify[b][i].top5list[4].m_pcnt;
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
            task_to_collect->flag_x100_csf_locate_inferred = lpLocation100xCSF->infer(task_to_collect->images, task_to_collect->result_x100_csf_locate);
            // NEW: 定位完成后，构建分片并通知分类线程池
            enqueue_csf_classify_jobs(task_to_collect);
            // cv_infer_csf_classify.notify_one();
        }
    }

    void enqueue_csf_classify_jobs(const std::shared_ptr<X100CSFTask>& task) {
        if (!task) return;
        bool expected = false;
        if (!task->csf_cls_in_progress.compare_exchange_strong(expected, true)) return;

        std::vector<CSFCellRef> flat;
        for (size_t i = 0; i < task->result_x100_csf_locate.size(); ++i) {
            const auto& loc = task->result_x100_csf_locate[i];
            for (int c = 0; c < loc.boxes.rows; ++c) flat.push_back({int(i), c});
        }
        std::cout << "定位细胞总数: " << flat.size() << std::endl;
        task->csf_total_cells = (int)flat.size();
        task->csf_remaining_cells = (int)flat.size();

        if (flat.empty()) {
            task->flag_x100_csf_classify_inferred = true;
            std::lock_guard<std::mutex> g(task_mutex);
            cv_csf_result.notify_one();
            task->csf_cls_in_progress = false;
            return;
        }

        task->result_x100_csf_classify.clear();
        task->result_x100_csf_classify.resize(task->result_x100_csf_locate.size());
        for (size_t i = 0; i < task->result_x100_csf_locate.size(); ++i)
            task->result_x100_csf_classify[i].resize(task->result_x100_csf_locate[i].boxes.rows);

        int k = calc_concurrency((int)flat.size());
        int per = (int)flat.size() / k, rem = (int)flat.size() % k;
        std::cout << "k : " << k << "  per: " << per << std::endl;
        {
            std::lock_guard<std::mutex> lk(csf_job_mtx);
            int off = 0;
            for (int i = 0; i < k; ++i) {
                int take = per + (i < rem ? 1 : 0);
                CSFCellShardJob job;
                job.task = task;
                job.cells.insert(job.cells.end(), flat.begin()+off, flat.begin()+off+take);
                csf_job_queue.emplace_back(std::move(job));
                off += take;
            }
        }
        cv_csf_jobs.notify_all();
    }


    // NEW: 分类工作线程
    void csf_classify_worker(int worker_idx) {
        libOnnxMNISTCSF* myClassifier = cls_instances_[worker_idx].get(); // 独享实例
        while (true) {
            CSFCellShardJob job;
            {
                std::unique_lock<std::mutex> lk(csf_job_mtx);
                cv_csf_jobs.wait(lk, [&]{ return stop_all || !csf_job_queue.empty(); });
                if (stop_all) return;
                job = std::move(csf_job_queue.front());
                csf_job_queue.pop_front();
            }
            auto task = job.task;
            if (!task) continue;

            for (const auto& ref : job.cells) {
                const auto& loc = task->result_x100_csf_locate[ref.img_idx];
                const float* box = loc.boxes.ptr<float>(ref.cell_idx);
                cv::Rect cell_rect((int)box[0], (int)box[1], (int)(box[2]-box[0]), (int)(box[3]-box[1]));
                cell_rect = makeUpSquare(cell_rect, task->images[ref.img_idx]);
                cell_rect = safeRect(cell_rect, task->images[ref.img_idx]);
                if (cell_rect.empty()) { if (task->csf_remaining_cells.fetch_sub(1) == 1) finish_task(task); continue; }

                cv::Mat cell_mat = task->images[ref.img_idx](cell_rect);
                if (cell_mat.empty()) { if (task->csf_remaining_cells.fetch_sub(1) == 1) finish_task(task); continue; }

                std::vector<itmCellRcgz_x100CSF> out;
                myClassifier->infer(cell_mat, out);
                if (out.size() < 5) out.resize(5); // 防越界

                itmCellRcgz_x100CSF_image cellinfo;
                cellinfo.cellRect = cell_rect;
                cellinfo.top5list = std::move(out);
                task->result_x100_csf_classify[ref.img_idx][ref.cell_idx] = std::move(cellinfo);

                if (task->csf_remaining_cells.fetch_sub(1) == 1) finish_task(task);
            }
        }
    }

    void finish_task(const std::shared_ptr<X100CSFTask>& task) {
        task->flag_x100_csf_classify_inferred = true;
        task->csf_cls_in_progress = false;
        std::lock_guard<std::mutex> g(task_mutex);
        cv_csf_result.notify_one();
    }


    void start_csf_classify_workers(int n) {
        // 先创建实例，再启动线程
        cls_instances_.reserve(n);
        for (int i = 0; i < n; ++i) {
            cls_instances_.push_back(std::make_unique<libOnnxMNISTCSF>(gpu_id_));
        }
        csf_cls_threads.reserve(n);
        for (int i = 0; i < n; ++i) {
            csf_cls_threads.emplace_back([this, i]{ csf_classify_worker(i); });
        }
    }

    void stop_csf_classify_workers() {
        {
            std::lock_guard<std::mutex> lk(csf_job_mtx);
            stop_all = true;
        }
        cv_csf_jobs.notify_all();
        for (auto &th : csf_cls_threads) if (th.joinable()) th.join();
        csf_cls_threads.clear();
        cls_instances_.clear(); // 显式释放
    }




    // void infer_csf_classify_thread() {
    //     while (true) {
    //         std::shared_ptr<X100CSFTask> task_to_collect;
    //         {
    //             std::unique_lock<std::mutex> lock(task_mutex);
    //             cv_infer_csf_classify.wait(lock, [&] {
    //                 if (stop_all) return true;
    //                 for (auto &t : task_queue) {
    //                     if (!t->flag_x100_csf_classify_inferred && t->flag_x100_csf_locate_inferred) {
    //                         return true;
    //                     }
    //                 }
    //                 return false;
    //             });
    //             if (stop_all) break;

    //             for (auto it = task_queue.begin(); it != task_queue.end(); ++it) {
    //                 if (!(*it)->flag_x100_csf_classify_inferred && (*it)->flag_x100_csf_locate_inferred) {
    //                     task_to_collect = *it;
    //                     break;
    //                 }
    //             }
    //         }
    //         if(task_to_collect->result_x100_csf_locate.size() <= 0)
    //         {
    //             task_to_collect->flag_x100_csf_classify_inferred = true;
    //         }
    //         else
    //         {
    //             for(size_t i = 0; i < task_to_collect->result_x100_csf_locate.size(); i++)
    //             {
    //                 //每张采样图
    //                 std::vector<itmCellRcgz_x100CSF_image> outlist;
    //                 // cv::Mat src = task_to_collect->images[i] + 0;
    //                 for(size_t cell = 0; cell < task_to_collect->result_x100_csf_locate[i].boxes.rows; cell++)
    //                 {
    //                     itmCellRcgz_x100CSF_image cellinfos;
    //                     //每张采样图上每个细胞
    //                     std::vector<itmCellRcgz_x100CSF> out;
    //                     const float* box = task_to_collect->result_x100_csf_locate[i].boxes.ptr<float>(cell);
    //                     cv::Rect cell_rect = cv::Rect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    //                     cv::Rect rect_new = makeUpSquare(cell_rect, task_to_collect->images[i]);
    //                     cv::Mat cell_mat = (task_to_collect->images[i])(cell_rect);
    //                     if(cell_mat.empty())
    //                     {
    //                         std::cout << "图片数据空 rect_new ----> " << rect_new << std::endl;
    //                         continue; 
    //                     }
    //                     // // cv::imwrite("cell.jpg", cell_mat);
    //                     lpClassify100xCSF->infer(cell_mat, out);
    //                     out.resize(5); // 只保留前 5 个元素
    //                     cellinfos.cellRect = cell_rect;
    //                     cellinfos.top5list = out;
    //                     outlist.push_back(cellinfos);
    //                 }
    //                 task_to_collect->result_x100_csf_classify.push_back(outlist);

    //             }
    //             task_to_collect->flag_x100_csf_classify_inferred = true;
    //         }
    //         cv_csf_result.notify_one();
    //     }
    // }
    
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