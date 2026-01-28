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

class X100HaveClassifyOnnx: public PublicTRT
{
public:
	static inline const int INPUTC = 3;
	static inline const int INPUTH = 300;
	static inline const int INPUTW = 300;
	static inline const int BATCH = 1;
	static inline const int OUTPUTSIZE = 35;
	static inline const cv::Scalar mean_ = cv::Scalar(0.485, 0.456, 0.406);
	static inline const cv::Scalar std_ = cv::Scalar(0.229, 0.224, 0.225);
    X100HaveClassifyOnnx(int gpu_id)
    {
		initTRT();
        std::string enginePath = selectEnginePath(gpu_id, "x100_have_classify");
        if (!loadEngine(enginePath)) {
            sample::gLogError << "Failed to load engine: " << enginePath << std::endl;
        }	
    }
    bool infer(const cv::Mat& image, std::vector<itmCellRcgz_x100>& out)
    {
		std::vector<cv::Mat> uImgs;
		uImgs.push_back(image);
		std::vector<float> inputData = processInput(uImgs);
        
        // 调用基类执行推理
        if (!doInference({{"input.1", inputData}})) return false;

        // 获取输出
        float* ptr_data = mHostOutputs["1394"].data();

        postprocess(ptr_data, out);
        return true;
	}

private:
    std::vector<float> processInput(const std::vector<cv::Mat>& srcs)
    {    
		int len = BATCH * INPUTC * INPUTW * INPUTH;
        std::vector<float> chw(len);
		float* data = chw.data();

		for(int b = 0; b < BATCH; b++)
		{
			cv::Mat image;
			cv::resize(srcs[b], image, cv::Size(INPUTW,INPUTH));
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

	void postprocess(float* output, std::vector<itmCellRcgz_x100>& out) {
        double sum = 0.0;
        std::vector<itmCellRcgz_x100> results;

        for (int i = 0; i < OUTPUTSIZE; ++i) {
            float val = std::min(output[i], 32.0f); // 防止 exp 溢出
            float expVal = std::exp(val);
            results.push_back({i, (double)expVal});
            sum += expVal;
        }

        for (auto& item : results) {
            item.m_pcnt /= sum;
        }

        // 排序：从大到小
        std::sort(results.begin(), results.end(), [](const itmCellRcgz_x100& a, const itmCellRcgz_x100& b) {
            return a.m_pcnt > b.m_pcnt;
        });

        out = std::move(results);
    }
};

