#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include "common.hpp"
#include <condition_variable>
#include "libX40BigCellLoc.hpp"
#include "lib40XConstituencyScore.hpp"
#include "libCellAnalysis.hpp"
#include "libX40HaveCellLocate2.hpp"
#include <functional>
#include <chrono>
#define PI                  acos(-1)
#define INPUT_IMAGE_NUM     4
#define BLOCK_NUM           4

struct X40Task {
        WorkerSharedBuffer* buffer_;
        int slot_id_;
        std::vector<cv::Mat> images;
        std::vector<cv::Mat> resize_images;
        std::vector<cv::Mat> white_balance_images;
        int image_actual_num = INPUT_IMAGE_NUM;

        bool flag_x40BigLocate_inferred = false;
        bool flag_cellAnalysis_inferred = false;
        bool flag_blockScore_inferred = false;
        bool flag_x40HaveLocate_inferred = false;

        std::vector<itmX40BigCellInfo> result_x40Big;
        std::vector<PicCellAnalysisResult> result_cellAnalysis;
        std::vector<itmCellRcgzConstituencyBigImg> result_blockScore;
        std::vector<itemX40HaveLocateInfo> result_x40HaveCellLocate;
    };

class X40TaskWorker {
public:
    using TaskSelector = std::function<bool(const std::shared_ptr<X40Task>&)>;
    using TaskProcessor = std::function<void(const std::shared_ptr<X40Task>&)>;
    using Notifier = std::function<void()>;

    X40TaskWorker(std::string model_name,
                  std::deque<std::shared_ptr<X40Task>>& tasks,
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
            std::shared_ptr<X40Task> task = nullptr;
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
    std::deque<std::shared_ptr<X40Task>>& task_queue_;
    std::mutex& mutex_;
    std::condition_variable& cv_;
    TaskSelector should_process_;
    TaskProcessor process_task_;
    Notifier notify_next_;
    std::atomic<bool>& stop_flag_;
};


class X40Main{

public:
    X40Main(int wid, int gpu_id) : worker_id(wid) {
        /*模型加载*/
        lpLocation40xBig = new X40BigCellLocateOnnx(gpu_id);
        if (!lpLocation40xBig) {
            std::cerr << "X40BigCellLocateOnnx 加载失败！" << std::endl;
        }

	    lpConstituency = new X40ConstituencyOnnx(gpu_id);
        if (!lpConstituency) {
            std::cerr << "X40ConstituencyOnnx 加载失败！" << std::endl;
        }

        lpCellAnalysis = new CellAnalysisOnnx(gpu_id);
        if (!lpCellAnalysis) {
            std::cerr << "CellAnalysisOnnx 加载失败！" << std::endl;
        }

        lpLocation40xHave = new CellLocateOnnx(gpu_id);
        if(!lpLocation40xHave){
            std::cerr << "CellLocateOnnx 加载失败！" << std::endl;
        }


        std::cout << "模型加载结束" << std::endl;

        stop_all = false;

        /*开启子线程*/
        start_workers();
        result_thread_ = std::thread(&X40Main::result_thread, this);
    }
    ~X40Main(){
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
        
        // /*释放模型指针*/
        // delete lpLocation40xBig;
        // lpLocation40xBig = nullptr;

        // delete lpConstituency;
        // lpConstituency = nullptr;

        // delete lpCellAnalysis;
        // lpCellAnalysis = nullptr;

    }
    /*添加任务到队列*/
    void add_x40_task(std::vector<cv::Mat>& imgs, WorkerSharedBuffer* buffer, int slot_id) {
        const int image_actual_num_local = static_cast<int>(imgs.size());
        std::vector<cv::Mat> padded = imgs;      // 避免在原imgs上反复push_back
        for (int i = image_actual_num_local; i < INPUT_IMAGE_NUM; ++i) {
            padded.push_back(padded[0]);         // padding
        }
        auto task = std::make_shared<X40Task>();
        task->images = padded;
        imageProcessing1(padded, task->resize_images);
        imageProcessing2(task->resize_images, task->white_balance_images);
        task->buffer_ = buffer;
        task->slot_id_ = slot_id;
        task->image_actual_num = image_actual_num_local; // 建议放到task里，不用全局
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            task_queue.emplace_back(task);
        }
        cv_infer.notify_all();
        // std::this_thread::yield();
    }

private:
    std::vector<std::thread> all_threads_;  // 存储所有X40TaskWorker线程
    std::deque<std::shared_ptr<X40Task>> task_queue;
    std::mutex task_mutex;
    std::condition_variable cv_infer;
    std::condition_variable cv_result;

    std::atomic<bool> stop_all = false;
    // int image_actual_num = INPUT_IMAGE_NUM;
    int worker_id;

    std::thread result_thread_;
    std::thread test_thread_;

    X40BigCellLocateOnnx *lpLocation40xBig;
    X40ConstituencyOnnx *lpConstituency;
    CellAnalysisOnnx *lpCellAnalysis;
    CellLocateOnnx *lpLocation40xHave;

    
    /*
        *************************************图片前处理*********************************
    */
    void imageProcessing1(vector<cv::Mat> uPicMatlist, vector<cv::Mat>& outPicMatlist)
    {
        outPicMatlist.clear();
        for(size_t i = 0; i < uPicMatlist.size(); i++)
        {
            cv::Mat src;
            cv::resize(uPicMatlist[i], src, cv::Size(612, 512));
            outPicMatlist.push_back(src);
        }
    }
    void imageProcessing2(vector<cv::Mat> uPicMatlist, vector<cv::Mat>& outPicMatlist)
    {
        const int inputH = 512;
        const int inputW = 640;

        outPicMatlist.clear();
        for(size_t i = 0; i < uPicMatlist.size(); i++)
        {
            cv::Mat uImg = uPicMatlist[i] + 0;
            cv::Mat gray;
            cv::cvtColor(uImg, gray, cv::COLOR_BGR2GRAY);
            cv::Scalar white_mean_gray = mean(gray);
            if (white_mean_gray[0] < 160)
            {
                cv::Mat flat = gray.reshape(1, 1) + 0;
                cv::sort(flat, flat, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
                int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * 0.85));
                cv::Mat white_mask = gray >= highval;
                cv::Scalar white_mean = mean(uImg, white_mask);

                std::vector<cv::Mat> channels;
                cv::split(uImg, channels);
                uImg.convertTo(uImg, CV_32FC3);
                channels.at(0) = channels.at(0) * (float(250) / white_mean[0]);
                channels.at(1) = channels.at(1) * (float(250) / white_mean[1]);
                channels.at(2) = channels.at(2) * (float(250) / white_mean[2]);
                cv::merge(channels, uImg);
                uImg.convertTo(uImg, CV_8UC3);
            }

            cv::Mat image(inputH, inputW, CV_8UC3, cv::Scalar(230, 230, 230));
            image(cv::Rect(0, 0, uImg.cols, uImg.rows)) = uImg + 0;
            outPicMatlist.push_back(image);
        }
    }

    // void filter_cell_by_center_point(std::vector<cv::Point> cellCenterPoints, cv::Mat b, std::vector<cv::Point>& new_out)
    // {
    //     for(size_t i = 0; i < cellCenterPoints.size(); i++)
    //     {
    //         int p_x = cellCenterPoints[i].x;
    //         int p_y = cellCenterPoints[i].y;
    //         if(b.at<uchar>(p_y, p_x) > 0)
    //         {
    //             new_out.push_back(cellCenterPoints[i]);
    //         }
    //     }
    //     return ;
    // }

    // void filter_cell_by_type(itemX40HaveLocateInfo result_x40HaveCellLocate, std::vector<cv::Point>& new_out)
    // {
    //     for(size_t i = 0; i < result_x40HaveCellLocate.cellrects.size(); i++)
    //     {
    //         int type = result_x40HaveCellLocate.types[i];
    //         if(type == 0)
    //         {
    //             int c_x = result_x40HaveCellLocate.cellrects[i].x + result_x40HaveCellLocate.cellrects[i].width / 2;
    //             int c_y = result_x40HaveCellLocate.cellrects[i].y + result_x40HaveCellLocate.cellrects[i].height / 2;
    //             new_out.push_back(cv::Point(c_x, c_y));
    //         }
    //     }
    //     return ;
    // }

    void filter_cell_by_type(itemX40HaveLocateInfo result_x40HaveCellLocate, cv::Mat b_image, cv::Mat image, std::vector<HaveCellResultInfos>& new_out)
    {
        for (int i = 0; i < result_x40HaveCellLocate.boxes.rows; ++i) {
            int cellType = result_x40HaveCellLocate.labels[i];

            if(cellType == 0)
            {
                const float* b = result_x40HaveCellLocate.boxes.ptr<float>(i);
                int x1 = b[0];
                int y1 = b[1];
                int x2 = b[2];
                int y2 = b[3];
                int c_x = (x1 + x2) / 2;
                int c_y = (y1 + y2) / 2;

                // /*画细胞*/
                // cv::circle(image, cv::Point(c_x, c_y), 5, cv::Scalar(0, 0, 255), -1);  
                // putText(image, to_string(cellType),
                //     cv::Point(c_x, c_y),                  // 文字位置
                //     FONT_HERSHEY_SIMPLEX,            // 字体
                //     1.5,                             // 字体大小
                //     Scalar(0, 255, 0),               // 绿色
                //     2,                               // 粗细
                //     LINE_AA);                        // 抗锯齿
                // cv::imwrite("out1.jpg", image);
                // cv::imwrite("b.jpg", b_image);
          
                // if(b_image.at<uchar>(c_y / 4, c_x / 4) > 0)
                // new_out.push_back(cv::Point(c_x, c_y));
                HaveCellResultInfos info;
                info.x1 = x1;
                info.y1 = y1;
                info.x2 = x2;
                info.y2 = y2;
                info.score = result_x40HaveCellLocate.scores[i];
                new_out.push_back(info);
            }
            
        }
    }
    /*
        ******************************计算区域评分**********************************
    */

    // 九宫格等级概率计算
	float nineClass_prod(float nineGrid_score)
	{
		float nineClass_v = 0.0f;
		if (nineGrid_score <= 64 && nineGrid_score > 45)
			nineClass_v = 8.5f * nineGrid_score * 0.001f;

		else if (nineGrid_score <= 45 && nineGrid_score > 24)
			nineClass_v = 5.5f * nineGrid_score * 0.001f;
		else
			nineClass_v = 0.16875f * nineGrid_score * 0.001f;

		return nineClass_v;
	}
	//有核细胞总面积归一化
	double total_MinMaxScaler(int x)
	{
		double data_min = 78.0;
		double data_max = 5604.0;
		double feature_range[2] = { 0.08, 5.6 };
		double scale = (feature_range[1] - feature_range[0]) / (data_max - data_min);
		double min_ = feature_range[0] - data_min * scale;
		double totalYouhe_y = scale * x + min_;
		return totalYouhe_y;
	}
	//有核细胞总面积概率计算: 服从卡方分布
	double totalArea_chi(double totalArea_scaler)
	{
		double k = 4.0;
		double scale = 0.55;
		totalArea_scaler = totalArea_scaler / scale;
		double a = 1 / ((pow(2, k / 2)) * 1);
		double b = pow(totalArea_scaler, (k / 2 - 1));
		double c = exp(-totalArea_scaler / 2);
		double totalArea_v = (a * b * c) / scale;
		return totalArea_v;
	}

	double totalArea_prod(int totalArea_Nucleated)
	{
		double totalArea_scaler = total_MinMaxScaler(totalArea_Nucleated);
		double totalArea_v = totalArea_chi(totalArea_scaler);
		if (totalArea_v < 0.0257)
			return  0.0257 + 0.1;

		else
			return totalArea_v + 0.1;
	}
	//红细胞面积归一化
	double red_MinMaxScaler(int x)
	{
		int data_min = 3076;
		int data_max = 17786;
		double feature_range[2] = { 3.0, 18.0 };
		double scale = (feature_range[1] - feature_range[0]) / (data_max - data_min);
		double min_ = feature_range[0] - data_min * scale;
		double red_y = scale * x + min_;
		return red_y;
	}
	// 红细胞总面积概率计算: 服从高斯分布
	double redArea_gauss(double redArea_scaler)
	{
		double loc = 9.120618882524237 - 0.5;
		double scale = 4.886065801133071 - 2.5;
		double a = (pow((redArea_scaler - loc), 2)) / (2 * pow(scale, 2));
		double Red_area_v = (1 / (scale * sqrt(2 * PI))) * exp(-a);
		return Red_area_v;
	}
	double redArea_prod(int Red_area)
	{

		double redArea_scaler = red_MinMaxScaler(Red_area);
		double Red_area_v = redArea_gauss(redArea_scaler);

		if (Red_area_v < 0.01)
			return 0.01 + 0.15;
		else
			return Red_area_v + 0.15;
	}
	/*计算九宫格分值*/
	float logPrior(float nineClass_v, int totalArea_v, int Red_area_v)
	{
		float logPrior_num = log(nineClass_prod(nineClass_v)) + log(float(totalArea_prod(totalArea_v))) + log(float(redArea_prod(Red_area_v)));
		return logPrior_num;
	}

    void confirm_block_grade_and_score(itmCellRcgzConstituencyBigImg blockScoreInfo, int &out_grade, float &out_score)
    {
        int block_top1 = blockScoreInfo.uBigData[0].m_type;
        int class_dict[7] = {64, 32, 16, 8, 4, 2, 1};
        if(block_top1 == 0)
        {
            float score_temp = 0.0f;
            for(size_t j = 0; j < blockScoreInfo.uBigData.size(); j++)
            {
                score_temp += blockScoreInfo.uBigData[j].m_pcnt * class_dict[blockScoreInfo.uBigData[j].m_type];
            }
            out_grade = 0;
            out_score = score_temp;
        }
        else if(block_top1 == 1)
        {
            float score_temp = 0.0f;
            for(size_t j = 0; j < blockScoreInfo.uBigData.size(); j++)
            {
                score_temp += blockScoreInfo.uBigData[j].m_pcnt * class_dict[blockScoreInfo.uBigData[j].m_type];
            }
            out_grade = 1;
            out_score = score_temp;
        }
        else if(block_top1 == 2)
        {
            float score_temp = 0.0f;
            for(size_t j = 0; j < blockScoreInfo.uBigData.size(); j++)
            {
                score_temp += blockScoreInfo.uBigData[j].m_pcnt * class_dict[blockScoreInfo.uBigData[j].m_type];
            }
            out_grade = 2;
            out_score = score_temp;
        }
        else if(block_top1 == 3)
        {
            out_grade = 3;
            out_score = class_dict[3];
        }
        else if(block_top1 == 4)
        {
            out_grade = 4;
            out_score = class_dict[4];
        }
        else if(block_top1 == 5)
        {
            out_grade = 5;
            out_score = class_dict[5];
        }
        else
        {
            out_grade = 6;
            out_score = class_dict[6];
        }
        return;
    }

     /*
        ******************************数据整合**********************************
    */
    void data_merge(std::shared_ptr<X40Task> &task)
    {
        {
            WorkerDataBlock &block = task->buffer_->blocks[task->slot_id_];
            for(int b = 0; b < task->image_actual_num; b++)
            {
                TaskDataBlock &result_ = block.task_batch_[b];
                std::vector<cv::Mat> channels;
                cv::split(task->result_cellAnalysis[b].uOutPic, channels);
                cv::Mat b_image = channels.at(0); //白细胞
                cv::Mat r_image = channels.at(2); //红细胞
                
                b_image = b_image > 200;
                r_image = r_image > 128;

                // //过滤非有核细胞
                // std::vector<cv::Point> X40HaveCellLocateOutFilter;
                std::vector<HaveCellResultInfos> X40HaveCellLocateOutFilter;
                // filter_cell_by_center_point(task->result_cellAnalysis[b].cellCenterPoints, b_image, X40HaveCellLocateOutFilter);
                filter_cell_by_type(task->result_x40HaveCellLocate[b], b_image, task->images[b], X40HaveCellLocateOutFilter);

                //计算区域分值
                int len = BLOCK_NUM;
                for(int bl = b *len; bl < (b + 1) * len; bl++)
                {
                    cv::Rect block_rect = task->result_blockScore[bl].uBigImg;
                    int b_area = cv::countNonZero(b_image(block_rect));
                    int r_area = cv::countNonZero(r_image(block_rect));
                    //确定区域等级和分值
                    int grade = 0;
                    float score = 0.0f;
                    confirm_block_grade_and_score(task->result_blockScore[bl], grade, score);
                    score = logPrior(score, b_area, r_area);

                    result_.result.imageResultInfos.areaScoreInfo[bl % len].x = block_rect.x;
                    result_.result.imageResultInfos.areaScoreInfo[bl % len].y = block_rect.y;
                    result_.result.imageResultInfos.areaScoreInfo[bl % len].w = block_rect.width;
                    result_.result.imageResultInfos.areaScoreInfo[bl % len].h = block_rect.height;
                    result_.result.imageResultInfos.areaScoreInfo[bl % len].score = score;
                    result_.result.imageResultInfos.areaScoreInfo[bl % len].grade = grade;
                }
                //有核细胞信息赋值
                result_.result.imageResultInfos.haveCellCenterPointsSize = X40HaveCellLocateOutFilter.size();
                for(size_t i = 0; i < X40HaveCellLocateOutFilter.size(); i++)
                {
                    // result_.result.imageResultInfos.haveCellCenterPoints[i].c_x = X40HaveCellLocateOutFilter[i].x;
                    // result_.result.imageResultInfos.haveCellCenterPoints[i].c_y = X40HaveCellLocateOutFilter[i].y;
                    result_.result.imageResultInfos.haveCellCenterPoints[i] = X40HaveCellLocateOutFilter[i];
                    // /*画细胞*/
                    // cv::circle(task->images[b], X40HaveCellLocateOutFilter[i], 5, cv::Scalar(0, 0, 255), -1);  
                    // cv::imwrite("out.jpg", task->images[b]);
                }
                //巨核细胞信息赋值
                result_.result.imageResultInfos.bigCellRectsSize = task->result_x40Big[b].bigCellInfo.size();
                for(size_t i = 0; i < task->result_x40Big[b].bigCellInfo.size(); i++)
                {
                    result_.result.imageResultInfos.bigCellRects[i].x = task->result_x40Big[b].bigCellInfo[i].x * 4;
                    result_.result.imageResultInfos.bigCellRects[i].y = task->result_x40Big[b].bigCellInfo[i].y * 4; 
                    result_.result.imageResultInfos.bigCellRects[i].w = task->result_x40Big[b].bigCellInfo[i].width * 4;
                    result_.result.imageResultInfos.bigCellRects[i].h = task->result_x40Big[b].bigCellInfo[i].height * 4; 
                    result_.result.imageResultInfos.bigCellRects[i].rate = task->result_x40Big[b].bigCellRate[i]; 
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
    void start_workers() {
        all_threads_.emplace_back([this]() {
            X40TaskWorker worker(
                "Infer-X40BigLocate",
                task_queue, task_mutex, cv_infer,
                [](const std::shared_ptr<X40Task>& t) {
                    return !t->flag_x40BigLocate_inferred;
                },
                [this](const std::shared_ptr<X40Task>& t) {
                    t->flag_x40BigLocate_inferred = lpLocation40xBig->infer(t->resize_images, t->result_x40Big);
                },
                [&]() { cv_result.notify_one(); },
                stop_all
            );
            worker();
        });

        all_threads_.emplace_back([this]() {
            X40TaskWorker worker(
                "Infer-CellAnalysis",
                task_queue, task_mutex, cv_infer,
                [](const std::shared_ptr<X40Task>& t) {
                    return !t->flag_cellAnalysis_inferred;
                },
                [this](const std::shared_ptr<X40Task>& t) {
                    t->flag_cellAnalysis_inferred = lpCellAnalysis->infer(t->white_balance_images, t->result_cellAnalysis);
                },
                [&]() { cv_result.notify_one(); },
                stop_all
            );
            worker();
        });

        all_threads_.emplace_back([this]() {
            X40TaskWorker worker(
                "Infer-x40HaveLocate",
                task_queue, task_mutex, cv_infer,
                [](const std::shared_ptr<X40Task>& t) {
                    return !t->flag_x40HaveLocate_inferred;
                },
                [this](const std::shared_ptr<X40Task>& t) {
                    t->flag_x40HaveLocate_inferred = lpLocation40xHave->infer(t->images, t->result_x40HaveCellLocate);
                },
                [&]() { cv_result.notify_one(); },
                stop_all
            );
            worker();
        });

        all_threads_.emplace_back([this]() {
            X40TaskWorker worker(
                "Infer-BlockScore",
                task_queue, task_mutex, cv_infer,
                [](const std::shared_ptr<X40Task>& t) {
                    return !t->flag_blockScore_inferred;
                },
                [this](const std::shared_ptr<X40Task>& t) {
                    t->flag_blockScore_inferred = lpConstituency->infer(t->resize_images, t->result_blockScore);
                },
                [&]() { cv_result.notify_one(); },
                stop_all
            );
            worker();
        });
    }

    
    void result_thread() {
        while (true) {
            std::shared_ptr<X40Task> task_to_collect;
            {
                std::unique_lock<std::mutex> lock(task_mutex);
                /*找到一个完成所有模型计算的任务*/
                cv_result.wait(lock, [&] {
                    if (stop_all) return true;
                    for (auto &t : task_queue) {

                        if (t->flag_x40BigLocate_inferred &&
                            t->flag_cellAnalysis_inferred &&
                            t->flag_blockScore_inferred &&
                            t->flag_x40HaveLocate_inferred) {
                            return true;
                        }
                    }
                    return false;

                });
                if (stop_all) break;

                for (auto it = task_queue.begin(); it != task_queue.end(); ++it) {
                    if ((*it)->flag_x40BigLocate_inferred 
                        && (*it)->flag_cellAnalysis_inferred
                        && (*it)->flag_blockScore_inferred
                        && (*it)->flag_x40HaveLocate_inferred) {
                        task_to_collect = *it;
                        task_queue.erase(it);
                        // std::cout << "task_queue.size()---->>> " << task_queue.size() << std::endl;
                        break;
                    }
                }
            }
            data_merge(task_to_collect);
        }
    }
};