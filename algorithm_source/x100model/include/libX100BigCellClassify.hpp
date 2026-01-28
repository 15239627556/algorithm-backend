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

class X100BigClassifyOnnx: public PublicTRT
{
public:
	static inline const int INPUTC = 3;
	static inline const int INPUTH = 128;
	static inline const int INPUTW = 128;
	static inline const int OUTPUTSIZE = 14; 
	static inline const int BATCH = 1;
	static inline cv::Scalar mean_ = cv::Scalar(0.485, 0.456, 0.406);
	static inline cv::Scalar std_ = cv::Scalar(0.229, 0.224, 0.225);
    X100BigClassifyOnnx(int gpu_id)
    {
		initTRT();
        std::string enginePath = selectEnginePath(gpu_id, "x100_big_classify");
        if (!loadEngine(enginePath)) {
            sample::gLogError << "Failed to load engine: " << enginePath << std::endl;
        }	
    }
    bool infer(cv::Mat& image, std::vector<itmCellRcgz_x100>& outList)
    {
		std::vector<cv::Mat> uImgs;
		uImgs.push_back(image);
		std::vector<float> inputData = processInput(uImgs);
        
        // 调用基类执行推理
        if (!doInference({{"input.1", inputData}})) return false;

        // 获取输出
        float* ptr_data = mHostOutputs["226"].data();

        postprocess(ptr_data, outList);
        return true;
	}

private:
    std::vector<float> processInput(std::vector<cv::Mat>& srcs)
    {
		int len = BATCH * INPUTC * INPUTW * INPUTH;
        std::vector<float> chw(len);
		float* data = chw.data();

		for(int b = 0; b < BATCH; b++)
		{
			cv::Mat image;
			cv::resize(srcs[b], image, cv::Size(INPUTW, INPUTH));
			cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
			image.convertTo(image, CV_32FC3);
			image = image / 255.0;
			cv::subtract(image, mean_, image);
			cv::divide(image, std_, image);

			std::vector<cv::Mat> channels(INPUTC);
			for (int c = 0; c < INPUTC; ++c)
			{
				// 每个通道指向 chw 向量中对应的平面起始位置
				channels[c] = cv::Mat(INPUTH, INPUTW, CV_32FC1, data + b * INPUTC * INPUTH * INPUTW + c * INPUTH * INPUTW);
			}
			// 将 HWC 的 image 拆分并直接拷贝到 channels 指向的 chw 内存中
			cv::split(image, channels);
		}
		return chw;
	}
	//后处理
	void postprocess(float* output, std::vector<itmCellRcgz_x100>& outList) {
		// 1. 准备容器并预留空间（假设 OUTPUTSIZE 是类别总数）
		std::vector<itmCellRcgz_x100> results;
		results.reserve(OUTPUTSIZE); 

		double sum = 0.0;

		// 2. 计算 Exp 和 Sum (带防溢出保护)
		for (int i = 0; i < OUTPUTSIZE; ++i) {
			// 防止 exp 溢出，32.0f 是安全阈值
			float val = std::min(output[i], 32.0f); 
			float expVal = std::exp(val);
			
			itmCellRcgz_x100 itm;
			itm.m_type = i;
			itm.m_pcnt = static_cast<double>(expVal);
			results.push_back(itm);
			
			sum += expVal;
		}

		// 3. 执行 Softmax 归一化
		if (sum > 0.0) {
			for (auto& item : results) {
				item.m_pcnt /= sum;
			}
		}

		// 4. 排序：按置信度 (m_pcnt) 从大到小
		std::sort(results.begin(), results.end(), [](const itmCellRcgz_x100& a, const itmCellRcgz_x100& b) {
			return a.m_pcnt > b.m_pcnt;
		});

		// 5. 将结果移动或拷贝到输出列表
		// 如果只需要前 N 个结果，可以在这里 resize
		outList = std::move(results); 
	}
};

