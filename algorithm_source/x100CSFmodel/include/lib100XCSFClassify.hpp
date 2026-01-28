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

class libOnnxMNISTCSF: public PublicTRT
{
public:
	static inline const int OUTPUTSIZE = 12; 
	static inline const int BATCH = 32;
	static inline const int INPUTC = 3;
	static inline const int INPUTH = 64;
	static inline const int INPUTW = 64;		
	static inline const  cv::Scalar mean_ = cv::Scalar(0.485, 0.456, 0.406);
	static inline const  cv::Scalar std_ = cv::Scalar(0.229, 0.224, 0.225);

    libOnnxMNISTCSF(int gpu_id)
    {
		initTRT();
        std::string enginePath = selectEnginePath(gpu_id, "csf_classify");
        if (!loadEngine(enginePath)) {
            sample::gLogError << "Failed to load engine: " << enginePath << std::endl;
        }	
    }
    bool infer(std::vector<cellMatAndIndex>& imagelist, std::vector<std::vector<itmCellRcgz_x100CSF>>& outlist)
    {
		std::vector<float> inputData = processInput(imagelist);
        
        // 调用基类执行推理
        if (!doInference({{"input.1", inputData}})) return false;

        // 获取输出
        float* ptr_data = mHostOutputs["993"].data();

        postprocess(ptr_data, outlist);
        return true;
	}

private:
    std::vector<float>  processInput(const std::vector<cellMatAndIndex>& srclist)
    {	    
	    int len = BATCH * INPUTC * INPUTW * INPUTH;
        std::vector<float> chw(len);
		float* data = chw.data();

		for(size_t b = 0; b < BATCH; b++)
		{
			cv::Mat image;
			cv::resize(srclist[b].cellMat, image, cv::Size(INPUTW,INPUTH));
			cv::cvtColor(image, image,  cv::COLOR_BGR2RGB);

			image.convertTo(image, CV_32FC3);
			image = image/255.0;
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
	void postprocess(float* output, std::vector<std::vector<itmCellRcgz_x100CSF>>& outlist) {
		for (int b = 0; b < BATCH; b++) {
			// 1. 定位当前 Batch 的起始指针
			float* currentOutput = output + (b * OUTPUTSIZE);
			
			// 2. 准备容器并预留空间
			std::vector<itmCellRcgz_x100CSF> results;
			results.reserve(OUTPUTSIZE);

			double sum = 0.0;

			// 3. 计算 Exp 和 Sum (防溢出处理)
			for (int i = 0; i < OUTPUTSIZE; ++i) {
				// 使用 std::min 防止 exp 溢出，避免了 rand() 的不确定性
				float val = std::min(currentOutput[i], 32.0f);
				float expVal = std::exp(val);
				
				itmCellRcgz_x100CSF itm;
				itm.m_type = i; // 类别索引 0 ~ OUTPUTSIZE-1
				itm.m_pcnt = static_cast<double>(expVal);
				results.push_back(itm);
				
				sum += expVal;
			}

			// 4. 执行 Softmax 归一化
			if (sum > 0.0) {
				for (auto& item : results) {
					item.m_pcnt /= sum;
				}
			}

			// 5. 排序：按置信度从大到小
			std::sort(results.begin(), results.end(), [](const itmCellRcgz_x100CSF& a, const itmCellRcgz_x100CSF& b) {
				return a.m_pcnt > b.m_pcnt;
			});

			// 6. 截断只保留前 5 个元素 (Top-5)
			if (results.size() > 5) {
				results.resize(5);
			}

			// 7. 将当前 Batch 的结果存入总列表
			outlist.push_back(std::move(results));
		}
	}
};
