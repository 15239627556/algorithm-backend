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
#include "publicTRT.hpp"

class EnhanceOnnx: public PublicTRT
{
public:
	static inline const int BATCH = 1;
	static inline const int INPUTC = 3;
	static inline const int INPUTH = 1024;
	static inline const int INPUTW = 1224;

	EnhanceOnnx(int gpu_id)
	{
		initTRT();
        std::string enginePath = selectEnginePath(gpu_id, "x40_enhance");
        if (!loadEngine(enginePath)) {
            sample::gLogError << "Failed to load engine: " << enginePath << std::endl;
        }		
	}

	bool infer(const cv::Mat uImg, cv::Mat& uOutImg)
	{
		std::vector<cv::Mat> uImgs;
		uImgs.push_back(uImg);
		std::vector<float> inputData = processInput(uImgs);
        
        // 调用基类执行推理
        if (!doInference({{"x", inputData}})) return false;

        // 获取输出
        float* ptr_data = mHostOutputs["1448"].data();

        postprocess(ptr_data, uOutImg);
        return true;
	}


private:
	std::vector<float> processInput(const std::vector<cv::Mat> uImgs)
	{
		int len = BATCH * INPUTC * INPUTW * INPUTH;
        std::vector<float> chw(len);
		float* data = chw.data();

		for(int b = 0; b < BATCH; b++)
		{
			cv::Mat uImg;
			cv::resize(uImgs[b], uImg, cv::Size(), .5, .5);
			cv::Mat rgb, imgF32;
			cv::cvtColor(uImg, rgb, cv::COLOR_BGR2RGB);
			rgb.convertTo(imgF32, CV_32FC3);
			imgF32 = imgF32 / 255.0;

			std::vector<cv::Mat> channels(INPUTC);
			for (int c = 0; c < INPUTC; ++c)
			{
				// 每个通道指向 chw 向量中对应的平面起始位置
				channels[c] = cv::Mat(INPUTH, INPUTW, CV_32FC1, data + b * INPUTC * INPUTH * INPUTW + c * INPUTH * INPUTW);
			}
			// 将 HWC 的 image 拆分并直接拷贝到 channels 指向的 chw 内存中
			cv::split(imgF32, channels);
		}
		return chw;
	}

	//后处理
	void postprocess(float* output, cv::Mat& out)
	{
		// 1. 严格对应模型的输出尺寸
		const int outH = 2048;
		const int outW = 2448;
		const size_t planeSize = static_cast<size_t>(outH) * outW; // 单个通道的大小

		if (output == nullptr) {
			std::cerr << "Error: Output pointer is null!" << std::endl;
			return;
		}

		// 2. 按照 CHW 格式从 output 缓冲区提取数据
		// 假设模型输出顺序是 R, G, B (索引 0, 1, 2)
		// OpenCV 需要 BGR 顺序
		cv::Mat channelR(outH, outW, CV_32FC1, output + 0 * planeSize);
		cv::Mat channelG(outH, outW, CV_32FC1, output + 1 * planeSize);
		cv::Mat channelB(outH, outW, CV_32FC1, output + 2 * planeSize);

		std::vector<cv::Mat> channels = { channelB, channelG, channelR };
		
		cv::Mat merged;
		cv::merge(channels, merged);

		// 3. 数值缩放与溢出保护
		// 很多增强模型输出已经是 0-1 范围，乘以 255 转为 8位图
		// 如果模型输出本身就是 0-255，则去掉 255.0 参数
		merged.convertTo(out, CV_8UC3, 255.0);

		// // 4. 调试：如果还是黑图，打印一下原始数据的最大值
		// double minVal, maxVal;
		// cv::minMaxLoc(channelR, &minVal, &maxVal);
		// std::cout << "Debug Output Range: [" << minVal << ", " << maxVal << "]" << std::endl;
	}
};

