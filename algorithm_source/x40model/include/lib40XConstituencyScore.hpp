#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logger.h"


typedef struct
{
	int m_type;
	double m_pcnt;
}itmCellRcgzConstituency;

struct  itmCellRcgzConstituencyBigImg
{
	cv::Rect uBigImg;
	std::vector<itmCellRcgzConstituency> uBigData;
};

class X40ConstituencyOnnx : public PublicTRT 
{
public:
	static inline const int PICNUM = 4; //接口输入的图片数量
	static inline const int BATCH = 16; //模型输入图片数量
	static inline const int INPUTW = 224; //模型输入图片W
	static inline const int INPUTH = 224; //模型输入图片H
	static inline const int INPUTC = 3; //模型输入图片通道数量
	static inline const int OUTPUTNUM = 7; //模型输出类别数量
	static inline const cv::Scalar mean_ = cv::Scalar(0.485, 0.456, 0.406);
	static inline const cv::Scalar std_ = cv::Scalar(0.229, 0.224, 0.225);

	X40ConstituencyOnnx(int gpu_id)
	{
		initTRT();

		std::string enginePath = selectEnginePath(gpu_id, "x40_constituency");

        if (!loadEngine(enginePath)) {
            sample::gLogError << "Failed to load engine: " << enginePath << std::endl;
        }
	}

	bool infer(std::vector<cv::Mat> uImgs, std::vector<itmCellRcgzConstituencyBigImg>& out)
	{
		// 预处理
        std::vector<float> inputData = processInput(uImgs, out);
        
        // 组织输入 (即使只有一个输入也用 Map)
        std::unordered_map<std::string, std::vector<float>> inputs = {{"input.1", inputData}};

        // 推理
        if (!doInference(inputs)) return false;

        // 直接从基类的 mHostOutputs 中取数据，名字对应 ONNX 的 Output Name
        float* prt_data = mHostOutputs["533"].data();

        postprocess(prt_data, out);
		return true;
	}

private:
	std::vector<float>  processInput(const std::vector<cv::Mat>& srcList, std::vector<itmCellRcgzConstituencyBigImg>& out)
	{	
		std::vector<cv::Mat> images;
		for (int m = 0; m < PICNUM; m++)
		{
			cv::Mat image;
			srcList[m].convertTo(image, CV_32FC3);
			image = image / 255.0;
			cv::subtract(image, mean_, image);
			cv::divide(image, std_, image);
			std::vector<cv::Rect> uImgRect;

			int uWIndex = srcList[m].cols / 612;
			int uHIndex = srcList[m].rows / 512;

			//图片分割
			int w_skip = 212;
			int h_skip = 200;
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					itmCellRcgzConstituencyBigImg item;
					images.push_back(image(cv::Rect(j*w_skip, (i*h_skip + 88), INPUTW, INPUTH)));
					item.uBigImg = cv::Rect(j*w_skip * uWIndex, (i*h_skip + 88) * uHIndex, INPUTW*uWIndex, INPUTH*uHIndex);
					out.push_back(item);
				}
			}
			
		}
		// HWC -> CHW
		int len = BATCH * INPUTC * INPUTW * INPUTH;
        std::vector<float> chw(len);
		float* data = chw.data();
		for (int b = 0; b < images.size() && b < BATCH; ++b)
		{
			// 分离通道: 将 HWC 拆分为三个单通道 Mat (C, H, W)
			std::vector<cv::Mat> channels(INPUTC);
			for (int c = 0; c < INPUTC; ++c)
			{
				// 每个通道指向 chw 向量中对应的平面起始位置
				channels[c] = cv::Mat(INPUTH, INPUTW, CV_32FC1, data + b * INPUTC * INPUTH * INPUTW + c * INPUTH * INPUTW);
			}
			// 将 HWC 的 image 拆分并直接拷贝到 channels 指向的 chw 内存中
			cv::split(images[b], channels);
		}
        
        return chw;
	}

	void postprocess(float* output, std::vector<itmCellRcgzConstituencyBigImg>& out)
	{
		for (int b = 0; b < BATCH; b++)
		{
			int offset = b * OUTPUTNUM;
			
			// 数值稳定的 Softmax (减去最大值防止 exp 溢出)
			float maxVal = output[offset];
			for (int i = 1; i < OUTPUTNUM; ++i) {
				if (output[offset + i] > maxVal) maxVal = output[offset + i];
			}

			float sum{ 0.0f };
			for (int i = 0; i < OUTPUTNUM; i++)
			{
				output[offset + i] = std::exp(output[offset + i] - maxVal);
				sum += output[offset + i];
			}

			// 构造当前 Batch 的结果集
			std::vector<itmCellRcgzConstituency> currentBatchData;
			currentBatchData.reserve(OUTPUTNUM); // 预分配空间

			for (int i = 0; i < OUTPUTNUM; i++)
			{
				float prob = output[offset + i] / sum;
				if (prob > 0.0f) { // 仅收集概率大于0的
					currentBatchData.push_back({i, (double)prob});
				}
			}

			// 使用高效排序代替插入排序 (O(N log N))
			std::sort(currentBatchData.begin(), currentBatchData.end(), 
				[](const itmCellRcgzConstituency& a, const itmCellRcgzConstituency& b) {
					return a.m_pcnt > b.m_pcnt; // 降序排列
				});

			// 直接填充到传入的 out 结构体中
			if (b < out.size()) {
				out[b].uBigData = std::move(currentBatchData);
			}
		}
	}
};
