#pragma once
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "opencv2/cudawarping.hpp"
#include <type_traits>
using namespace cv;
using namespace std;

// struct itemX40HaveLocateInfo
// {
//     std::vector<cv::Rect> cellrects;
//     std::vector<float> scores;
//     std::vector<int> types; //0-有核
// };
struct itemX40HaveLocateInfo {
    cv::Mat boxes;
    std::vector<float> scores;
    std::vector<int>   labels;//0-有核
};

class CellLocateOnnx
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	CellLocateOnnx(int gpu_id)
	{
		mParams.inputTensorNames.push_back("images");
		mParams.batchSize = 9;
		mParams.outputTensorNames.push_back("output0");
		mParams.dlaCore = -1;
		mParams.int8 = false;
		mParams.fp16 = false;
		initLibNvInferPlugins(nullptr, "");
        // cudaDeviceProp deviceProp;
		// cudaGetDeviceProperties(&deviceProp, gpu_id);
		// string diviceName = deviceProp.name;
        string diviceName = GPU_NAMES[gpu_id];
		size_t index_2080 = diviceName.find("2080");
		size_t index_3080 = diviceName.find("3080");
		size_t index_4070 = diviceName.find("4070");
		size_t index_4090 = diviceName.find("4090");
		std::string engine = "";
		if(index_2080 != string::npos)
			engine = "engines/2080/x40_have_locate_.trt";
		else if(index_3080 != string::npos)
			engine = "engines/3080/x40_have_locate_.trt";
		else if(index_4070 != string::npos)
			engine = "engines/4070/x40_have_locate_.trt";
		else if(index_4090 != string::npos)
			engine = "engines/4070/x40_have_locate_.trt";
		else
			std::cout << "cannot find correct trt" << std::endl;

		// std::string engine = "engines/x40_have_locate_.trt";
        // std::string engine = "engines/4070/x40_have_locate_.trt";
		std::ifstream engineFile(engine, std::ios::binary);
		if (!engineFile)
		{
			sample::gLogInfo << "Error opening engine file: " << engine << std::endl;
			return ;
		}
		engineFile.seekg(0, engineFile.end);
		long int fsize = engineFile.tellg();
		engineFile.seekg(0, engineFile.beg);

		std::vector<char> engineData(fsize);
		engineFile.read(engineData.data(), fsize);
		if (!engineFile)
		{
			sample::gLogInfo << "Error loading engine file: " << engine << std::endl;
			return ;
		}
		sample::gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kERROR);  // 设置日志级别
		mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
		mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
	}
	

	~CellLocateOnnx()
	{}
	
	bool infer(std::vector<cv::Mat> uImgs, std::vector<itemX40HaveLocateInfo>& uOutX40HaveCellLocates)
	{
		samplesCommon::BufferManager buffers(mEngine);
    	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
		if (!context)
		{
			return false;
		}
		// Read the input data into the managed buffers
		assert(mParams.inputTensorNames.size() == 1);

		//double time1 = static_cast<double>(cv::getTickCount());
		std::vector<float> rs;
        std::vector<cv::Point> pads;
		if (!processInput(buffers, uImgs, rs, pads))
		{
			return false;
		}
		// cudaSetDevice(device);
        buffers.copyInputToDevice();

		bool status = context->executeV2(buffers.getDeviceBindings().data());
		
		if (!status)
		{
			return false;
		}
		buffers.copyOutputToHost();
		const int batchSize = 4;
        const int class_num = 6;
		float* data = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
        int B = 4;                 // 你的 batch
        int A = 6;                 // 通道维（或 N），请以实际为准
        int N = 109395;            // box 数（或通道数），请以实际为准

        const float conf = 0.40f;
        const float iou  = 0.60f;
        const int   maxd = 2000;
        const bool  agn  = true;   // 与导出 onnx 的 agnostic_nms=True 一致

        cv::Mat pred_mat = makePredMatView(data, B, A, N);

        uOutX40HaveCellLocates = postprocess_no_nms_cpp(
            pred_mat,           // 预测
            conf,           // 置信度阈值
            iou,            // IoU 阈值
            maxd,           // 每图最多检测数
            agn,            // 类无关 NMS
            /*num_classes=*/2,
            /*apply_sigmoid=*/false,
            rs,
            pads,
            /*classes=*/nullptr
        );

        const auto& det = uOutX40HaveCellLocates[0];

        for (int r = 0; r < det.boxes.rows; ++r) {
            float score = det.scores[r];

            const float* b = det.boxes.ptr<float>(r); // [x1,y1,x2,y2]
            int label = det.labels[r];

            // 组装 "[x1, y1, x2, y2]" 字符串，box 一位小数
            std::ostringstream box_ss;
            box_ss << "["
                   << std::fixed << std::setprecision(1) << b[0] << ", "
                   << std::fixed << std::setprecision(1) << b[1] << ", "
                   << std::fixed << std::setprecision(1) << b[2] << ", "
                   << std::fixed << std::setprecision(1) << b[3] << "]";

            // 打印一行：文件名 -> Box: [...], Score: xx.xx, Label: x
            std::cout << "name"
                      << " -> Box: " << box_ss.str()
                      << ", Score: " << std::fixed << std::setprecision(2) << score
                      << ", Label: " << label
                      << '\n';
        }
        			
		return true;
	}


private:
	samplesCommon::OnnxSampleParams mParams; 
	std::shared_ptr<nvinfer1::IRuntime> mRuntime; 
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
	int mNumber{ 0 };             //!< The number to classify

    struct LetterboxResult {
        cv::Mat img;   // 处理后图像 (H x W x 3, uint8)
        float r;       // 等比缩放因子（标量）
        cv::Point pad; // 左上角 padding (left, top)
    };
    cv::Mat makePredMatView(float* pred, int B, int A, int N) {
        int sizes[3]  = {B, A, N};
        // 若为标准 C 连续布局（最后一维最连续），显式 steps 更安全：
        size_t steps[3] = {
            (size_t)A * N * sizeof(float),  // 步长：跨 batch
            (size_t)N * sizeof(float),      // 跨通道/第二维
            sizeof(float)                   // 跨最后一维
        };
        // 这只是一个“视图”，不接管 pred 的内存生命周期
        return cv::Mat(3, sizes, CV_32F, (void*)pred, steps);
    }
    // 计算两个 bbox（xyxy 格式）的 IoU
    // -----------------------------------------------------------------------------
    // 用途：NMS 中计算重叠度，决定是否抑制。
    // 参数：
    //   a, b  - 指向各自 4 个 float 的指针，依次是 [x1, y1, x2, y2]（左上到右下）。
    // 返回：
    //   IoU ∈ [0, 1]，并对并集加了 1e-7f 防止除零。
    // 注意：假定 x2>=x1 且 y2>=y1（若存在数值噪声，std::max(0, w/h) 已做保护）。
    static inline float iou_xyxy(const float* a, const float* b) {
        float ax1=a[0], ay1=a[1], ax2=a[2], ay2=a[3];
        float bx1=b[0], by1=b[1], bx2=b[2], by2=b[3];

        // 交集矩形左上与右下
        float ix1 = std::max(ax1, bx1);
        float iy1 = std::max(ay1, by1);
        float ix2 = std::min(ax2, bx2);
        float iy2 = std::min(ay2, by2);

        // 交集宽高与面积（负值取 0，表示无交集）
        float iw = std::max(0.0f, ix2 - ix1);
        float ih = std::max(0.0f, iy2 - iy1);
        float inter = iw * ih;

        // 并集 = 面积和 - 交集
        float areaA = std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);
        float areaB = std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);
        float uni = areaA + areaB - inter + 1e-7f; // 数值稳定性

        return inter / uni;
    }
    static std::vector<int> nms_singleclass(const cv::Mat& boxes_xyxy,
                                        const std::vector<float>& scores,
                                        float iou_thr,
                                        int max_det) {
        const int n = boxes_xyxy.rows;

        // 1) 按分数降序排序，记录索引
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                [&](int i, int j){ return scores[i] > scores[j]; });

        // 2) 贪心选择：遇到与已选 IoU>阈值的就抑制
        std::vector<int> keep;
        keep.reserve(std::min(max_det, n));
        std::vector<char> suppressed(n, 0);

        for (int _i = 0; _i < n && (int)keep.size() < max_det; ++_i) {
            int i = order[_i];
            if (suppressed[i]) continue;  // 已被抑制则跳过
            keep.push_back(i);
            const float* bi = boxes_xyxy.ptr<float>(i);

            // 和后续框两两比较 IoU，超过阈值的做抑制
            for (int _j = _i + 1; _j < n; ++_j) {
                int j = order[_j];
                if (suppressed[j]) continue;
                const float* bj = boxes_xyxy.ptr<float>(j);
                if (iou_xyxy(bi, bj) > iou_thr) suppressed[j] = 1;
            }
        }
        return keep;
    }
    static std::vector<int> yolov_like_nms_cpp(cv::Mat boxes_xyxy,           // copy
                                           std::vector<float> conf,
                                           std::vector<int> cls,
                                           float iou_thr,
                                           int max_det,
                                           bool agnostic,
                                           float max_wh = 7680.0f,
                                           int   max_nms = 30000,
                                           const std::vector<int>* classes = nullptr) {
        // (1) 按类别白名单过滤（如果提供了 classes）
        if (classes && !classes->empty()) {
            std::vector<int> keep_idx;
            keep_idx.reserve(cls.size());
            for (int i = 0; i < (int)cls.size(); ++i) {
                if (std::find(classes->begin(), classes->end(), cls[i]) != classes->end())
                    keep_idx.push_back(i);
            }
            // 根据 keep_idx 压缩 boxes/conf/cls
            cv::Mat nb((int)keep_idx.size(), 4, CV_32F);
            std::vector<float> nc; nc.reserve(keep_idx.size());
            std::vector<int>   nl; nl.reserve(keep_idx.size());
            for (int k = 0; k < (int)keep_idx.size(); ++k) {
                int i = keep_idx[k];
                boxes_xyxy.row(i).copyTo(nb.row(k));
                nc.push_back(conf[i]);
                nl.push_back(cls[i]);
            }
            boxes_xyxy = std::move(nb);
            conf = std::move(nc);
            cls  = std::move(nl);
        }

        int N = boxes_xyxy.rows;
        if (N == 0) return {};

        // (2) 预裁剪：若候选很多，仅保留分数 top-k（与 YOLO 的 max_nms 一致）
        if (N > max_nms) {
            std::vector<int> order(N);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + max_nms, order.end(),
                            [&](int i, int j){ return conf[i] > conf[j]; });

            cv::Mat nb(max_nms, 4, CV_32F);
            std::vector<float> nc; nc.reserve(max_nms);
            std::vector<int>   nl; nl.reserve(max_nms);
            for (int k = 0; k < max_nms; ++k) {
                int i = order[k];
                boxes_xyxy.row(i).copyTo(nb.row(k));
                nc.push_back(conf[i]);
                nl.push_back(cls[i]);
            }
            boxes_xyxy = std::move(nb);
            conf = std::move(nc);
            cls  = std::move(nl);
            N = boxes_xyxy.rows;
        }

        // (3) 类相关/类无关处理：
        //     - 类相关（agnostic=false）：对每个框的 x1,y1,x2,y2 全部加上 (class_id * max_wh)，
        //       这样不同类别的框被移到不同的坐标带中，互不抑制。
        if (!agnostic) {
            for (int i = 0; i < N; ++i) {
                float c = (float)cls[i] * max_wh;
                float* b = boxes_xyxy.ptr<float>(i);
                b[0]+=c; b[1]+=c; b[2]+=c; b[3]+=c;
            }
        }

        // (4) 调用单类贪心 NMS
        return nms_singleclass(boxes_xyxy, conf, iou_thr, max_det);
    }
    static std::vector<itemX40HaveLocateInfo> postprocess_no_nms_cpp(const cv::Mat& pred,
                                                    float conf_thr,
                                                    float iou_thr,
                                                    int   max_det,
                                                    bool  agnostic,
                                                    int   num_classes,
                                                    bool  apply_sigmoid,
                                                    std::vector<float> rs,
                                                    std::vector<cv::Point> pads,
                                                    const std::vector<int>* classes = nullptr) {
        // 基本形状与类型检查
        if (pred.dims != 3 || pred.type() != CV_32F) {
            throw std::runtime_error("pred must be 3D CV_32F");
        }
        const int B = pred.size[0];
        const int A = pred.size[1];
        const int N = pred.size[2];

        // (B,6,N) 视作 channels_first；否则要求 (B,N,6)
        const bool channels_first = (A == 6);
        if (!channels_first && N != 6)
            throw std::runtime_error("shape must be (B,6,N) or (B,N,6)");

        // 这里你的模型是 2 类：C = 6 - 4
        int C = 6 - 4;
        if (num_classes != C) num_classes = C;

        // 便捷访问 pred[b, a, n]
        auto at3 = [&](int i, int j, int k)->float {
            int idx[3] = {i,j,k};
            return pred.at<float>(idx);
        };

        std::vector<itemX40HaveLocateInfo> out;
        out.reserve(B);

        for (int b = 0; b < B; ++b) {
            // 拆出 xywh 与 per-class 分数
            cv::Mat boxes_xywh(N, 4, CV_32F);
            cv::Mat scores    (N, C, CV_32F);

            if (channels_first) {  // (B,6,N)
                for (int n = 0; n < N; ++n) {
                    float* bx = boxes_xywh.ptr<float>(n);
                    bx[0]=at3(b,0,n); bx[1]=at3(b,1,n);
                    bx[2]=at3(b,2,n); bx[3]=at3(b,3,n);
                    for (int c = 0; c < C; ++c) scores.at<float>(n,c)=at3(b,4+c,n);
                }
            } else {               // (B,N,6)
                for (int n = 0; n < N; ++n) {
                    float* bx = boxes_xywh.ptr<float>(n);
                    for (int k = 0; k < 4; ++k) { int idx[3]={b,n,k}; bx[k]=pred.at<float>(idx); }
                    for (int c = 0; c < C; ++c) { int idx[3]={b,n,4+c}; scores.at<float>(n,c)=pred.at<float>(idx); }
                }
            }

            // 若模型输出为 logits，这里做 sigmoid -> 概率
            if (apply_sigmoid) {
                for (int n=0;n<N;++n){
                    float* r = scores.ptr<float>(n);
                    for (int c=0;c<C;++c) r[c] = 1.0f/(1.0f+std::exp(-r[c]));
                }
            }

            // 单标签：每个候选取分数最大的类别（与 Ultralytics 默认 multi_label=False 一致）
            std::vector<float> conf(N);
            std::vector<int>   cls (N);
            for (int n = 0; n < N; ++n) {
                const float* r = scores.ptr<float>(n);
                int arg=0; float mx=r[0];
                for (int c=1;c<C;++c) if (r[c]>mx){mx=r[c]; arg=c;}
                conf[n]=mx; cls[n]=arg;
            }

            // 先做置信度阈值过滤，减少 NMS 工作量
            std::vector<int> keep_conf; keep_conf.reserve(N);
            for (int n=0;n<N;++n) if (conf[n]>=conf_thr) keep_conf.push_back(n);

            itemX40HaveLocateInfo dr;
            if (keep_conf.empty()) {
                // 无检测，返回空
                dr.boxes = cv::Mat(0,4,CV_32F);
                out.emplace_back(std::move(dr));
                continue;
            }

            // xywh -> xyxy，并收集对应的分数与类别
            int M = (int)keep_conf.size();
            dr.boxes.create(M,4,CV_32F);
            std::vector<float> cf; cf.reserve(M);
            std::vector<int>   cl; cl.reserve(M);

            for (int k=0;k<M;++k){
                int n = keep_conf[k];
                const float* wh = boxes_xywh.ptr<float>(n);
                float x=wh[0], y=wh[1], w=wh[2], h=wh[3];
                float* bb = dr.boxes.ptr<float>(k);
                bb[0]=x-w*0.5f; bb[1]=y-h*0.5f; bb[2]=x+w*0.5f; bb[3]=y+h*0.5f; // xywh -> xyxy
                cf.push_back(conf[n]);
                cl.push_back(cls[n]);
            }
            // scale_boxes(dr.boxes, rs[b], pads[b], 2448, 2048);
            // 调用 YOLO 风格 NMS（支持 agnostic、classes、max_nms 等行为）
            auto keep = yolov_like_nms_cpp(dr.boxes, cf, cl,
                                        iou_thr, std::min(max_det,M),
                                        agnostic, 7680.0f, 30000, classes);

            // 为了和 Python 对齐/便于回归测试：统一稳定排序
            // （按 score 降序；若分数相同，用 x1,y1,x2,y2 的字典序作为副排序键）
            std::vector<int> kk = keep;
            std::sort(kk.begin(), kk.end(), [&](int i, int j){
                if (cf[i] != cf[j]) return cf[i] > cf[j];
                const float* bi = dr.boxes.ptr<float>(i);
                const float* bj = dr.boxes.ptr<float>(j);
                if (bi[0]!=bj[0]) return bi[0]<bj[0];
                if (bi[1]!=bj[1]) return bi[1]<bj[1];
                if (bi[2]!=bj[2]) return bi[2]<bj[2];
                return bi[3]<bj[3];
            });

            // 输出最终保留结果
            itemX40HaveLocateInfo kept;
            kept.boxes.create((int)kk.size(),4,CV_32F);
            kept.scores.reserve(kk.size());
            kept.labels.reserve(kk.size());
            for (int t=0;t<(int)kk.size();++t){
                int k = kk[t];
                dr.boxes.row(k).copyTo(kept.boxes.row(t));
                kept.scores.push_back(cf[k]);
                kept.labels.push_back(cl[k]);
            }
            scale_boxes(kept.boxes, rs[b], pads[b], 2448, 2048);
            out.emplace_back(std::move(kept));
        }
        return out;
    }

    LetterboxResult letterbox(const cv::Mat& im,
                          cv::Size new_shape = cv::Size(1248, 1248), // 注意 OpenCV 的 Size 是 (W,H)
                          const cv::Scalar& color = cv::Scalar(114,114,114),
                          bool auto_shape = true,
                          bool scale_fill = false,
                          bool scaleup = true,
                          bool center = true,
                          int stride = 32,
                          int interpolation = cv::INTER_LINEAR) {
        assert(!im.empty());
        const int h = im.rows, w = im.cols;
        const int new_w = new_shape.width;
        const int new_h = new_shape.height;

        // 缩放因子
        float r = std::min(static_cast<float>(new_h) / static_cast<float>(h),
                        static_cast<float>(new_w) / static_cast<float>(w));
        if (!scaleup) r = std::min(r, 1.0f);

        // 缩放后未填充大小 (w', h')
        cv::Size new_unpad(std::lround(w * r), std::lround(h * r));

        float dw = static_cast<float>(new_w - new_unpad.width);
        float dh = static_cast<float>(new_h - new_unpad.height);

        if (scale_fill) {
            // 拉伸到目标大小（不保比例）
            new_unpad = cv::Size(new_w, new_h);
            dw = dh = 0.0f;
        } else if (auto_shape) {
            // 最小矩形：padding 与 stride 对齐
            dw = std::fmod(dw, static_cast<float>(stride));
            dh = std::fmod(dh, static_cast<float>(stride));
        }

        if (center) { dw *= 0.5f; dh *= 0.5f; }

        // 先 resize
        cv::Mat resized;
        if (new_unpad.width != w || new_unpad.height != h) {
            cv::resize(im, resized, new_unpad, 0, 0, interpolation);
        } else {
            resized = im.clone();
        }

        // 再 pad（与 Py 版 round(d±0.1) 对齐）
        const int top    = static_cast<int>(std::round(dh - 0.1f));
        const int bottom = static_cast<int>(std::round(dh + 0.1f));
        const int left   = static_cast<int>(std::round(dw - 0.1f));
        const int right  = static_cast<int>(std::round(dw + 0.1f));

        cv::Mat out;
        cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, color);

        return LetterboxResult{std::move(out), r, cv::Point(left, top)};
    }
    static void scale_boxes(cv::Mat& boxes_xyxy, float r, const cv::Point& pad,
                 int orig_w, int orig_h) {
        // boxes_xyxy: CV_32F, Nx4 (x1,y1,x2,y2)
        if (boxes_xyxy.empty()) return;
        CV_Assert(boxes_xyxy.type() == CV_32F && boxes_xyxy.cols == 4);

        // 回映射
        boxes_xyxy.col(0) = (boxes_xyxy.col(0) - (float)pad.x) / r; // x1
        boxes_xyxy.col(2) = (boxes_xyxy.col(2) - (float)pad.x) / r; // x2
        boxes_xyxy.col(1) = (boxes_xyxy.col(1) - (float)pad.y) / r; // y1
        boxes_xyxy.col(3) = (boxes_xyxy.col(3) - (float)pad.y) / r; // y2

        // 裁剪并保证 x2>=x1, y2>=y1
        const float W = static_cast<float>(orig_w);
        const float H = static_cast<float>(orig_h);
        for (int i = 0; i < boxes_xyxy.rows; ++i) {
            float& x1 = boxes_xyxy.at<float>(i, 0);
            float& y1 = boxes_xyxy.at<float>(i, 1);
            float& x2 = boxes_xyxy.at<float>(i, 2);
            float& y2 = boxes_xyxy.at<float>(i, 3);

            x1 = std::clamp(x1, 0.0f, W);
            x2 = std::clamp(x2, 0.0f, W);
            y1 = std::clamp(y1, 0.0f, H);
            y2 = std::clamp(y2, 0.0f, H);

            if (x2 < x1) x2 = x1;  // 防数值误差
            if (y2 < y1) y2 = y1;
        }
    }
	bool processInput(const samplesCommon::BufferManager& buffers, vector<cv::Mat> uImgs, std::vector<float>& rs, std::vector<cv::Point>& pads)
	{
		
		const int inputC = 3;
		const int inputH = 1056;
		const int inputW = 1248;
		const int batchSize = 4;
        float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
		for (int m = 0; m < batchSize; m++)
		{
            cv::Size new_shape(1248, 1248);
			auto lb = letterbox(uImgs[m], new_shape,
                        cv::Scalar(114,114,114),
                        true,
                        false,
                        true,
                        true,
                        32,
                        cv::INTER_LINEAR);
            rs.push_back(lb.r);
            pads.push_back(lb.pad);
            // BGR->RGB
            cv::Mat rgb;
            cv::cvtColor(lb.img, rgb, cv::COLOR_BGR2RGB);
            cv::Mat blob;
            rgb.convertTo(blob, CV_32FC3);
			blob = blob / 255.0;

            for (int i = 0; i < inputH; i++) {
				float* data = blob.ptr<float>(i);
				for (int j = 0; j < inputW; j++) {
					for (int k = 0; k < inputC; k++) {
                        // std::cout << i << " " << j << " " << k << std::endl;
						hostDataBuffer[m*inputH*inputW*inputC + k * inputW*inputH + i * inputW + j] = data[j*inputC + k];					
					}
				}
			}
		}
        
		return true;
	}
};