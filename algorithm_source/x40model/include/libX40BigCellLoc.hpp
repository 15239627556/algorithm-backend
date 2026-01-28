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

//x40巨核细胞定位结果
struct  itmX40BigCellInfo
{
	int index;
	std::vector<cv::Rect> bigCellInfo;
	std::vector<float> bigCellRate;
};

class X40BigCellLocateOnnx : public PublicTRT {
public:
	static inline const int BATCH = 4;
	static inline const int INPUTH = 512;
	static inline const int INPUTW = 640;
	static inline const int INPUTC = 3;

    X40BigCellLocateOnnx(int gpu_id) {
        initTRT();
        std::string enginePath = selectEnginePath(gpu_id, "x40_bigcell_locate");
        if (!loadEngine(enginePath)) {
            sample::gLogError << "Failed to load engine: " << enginePath << std::endl;
        }
    }

    bool infer(std::vector<cv::Mat> images, std::vector<itmX40BigCellInfo>& out) {
        std::vector<float> inputData = processInput(images);
        
        // 调用基类执行推理
        if (!doInference({{"input", inputData}})) return false;

        // 获取输出
        float* ptr_scores = mHostOutputs["cls_scores"].data();
        float* ptr_boxes  = mHostOutputs["boxes"].data();

        postprocess(ptr_scores, ptr_boxes, images, out);
        return true;
    }
private:
	//前处理
	std::vector<float> processInput(const std::vector<cv::Mat>& srclist)
	{
		int len = BATCH * INPUTC * INPUTW * INPUTH;
        std::vector<float> chw(len);
		float* data = chw.data();

		for (int m = 0; m < BATCH; m++)
		{
			cv::Mat src = srclist[m] + 0;
			cv::Mat image(INPUTH, INPUTW, CV_8UC3, cv::Scalar(114, 114, 114));
			image(cv::Rect(0, 0, src.cols, src.rows)) = src + 0;
			image.convertTo(image, CV_32FC3);

			std::vector<cv::Mat> channels(INPUTC);
			for (int c = 0; c < INPUTC; ++c)
			{
				// 每个通道指向 chw 向量中对应的平面起始位置
				channels[c] = cv::Mat(INPUTH, INPUTW, CV_32FC1, data + m * INPUTC * INPUTH * INPUTW + c * INPUTH * INPUTW);
			}
			// 将 HWC 的 image 拆分并直接拷贝到 channels 指向的 chw 内存中
			cv::split(image, channels);
		}
		return chw;
	}
	//后处理
	void postprocess(const float* rates, const float* rects, std::vector<cv::Mat> images, std::vector<itmX40BigCellInfo>& out)
	{
		for (int b = 0; b < BATCH; b++)
		{
			std::vector<cv::Rect> localBoxes;
			std::vector<float> localConfidences;
			int len = 6720;
			for (int i = b * len; i < (b + 1) * len; i++)
			{
				if (rates[i] > 0.45)
				{	
					localConfidences.push_back(rates[i]);
					localBoxes.push_back(cv::Rect(rects[i * 4] * images[b].cols / 612, rects[i * 4 + 1] * images[b].rows / INPUTH,
						rects[i * 4 + 2] * images[b].cols / 612 - rects[i * 4] * images[b].cols / 612,
						rects[i * 4 + 3] * images[b].rows / INPUTH - rects[i * 4 + 1] * images[b].rows / INPUTH));
				}
			}
			// NMS
			std::vector<int> nmsIndices;
			cv::dnn::NMSBoxes(localBoxes, localConfidences, 0.45f, 0.7f, nmsIndices);
			
			itmX40BigCellInfo tempInfo;
			tempInfo.index = b;
			tempInfo.bigCellRate = {};
			tempInfo.bigCellInfo = {};
			for (size_t idx = 0; idx < nmsIndices.size(); idx++) {
				size_t idx_ = nmsIndices[idx];
				
				tempInfo.bigCellRate.push_back(localConfidences[idx_]);
				//外扩细胞框
				int x_new = localBoxes[idx_].x - localBoxes[idx_].width / 2;
				int y_new = localBoxes[idx_].y - localBoxes[idx_].height / 2;
				int w_new = localBoxes[idx_].width * 2;
				int h_new = localBoxes[idx_].height * 2;
				cv::Rect cell_rect_new = cv::Rect(x_new, y_new, w_new, h_new);

				tempInfo.bigCellInfo.push_back(cell_rect_new & cv::Rect(0, 0, images[b].cols, images[b].rows));
				
			}
			out.push_back(tempInfo);
		}
		return;
	}
};