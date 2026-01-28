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


class X100BigLocateOnnx: public PublicTRT
{
public:
	static inline const int INPUTC = 3;
	static inline const int INPUTH = 256;
	static inline const int INPUTW = 320;
	static inline const int BATCH = 1;
	X100BigLocateOnnx(int gpu_id)
	{
		initTRT();
        std::string enginePath = selectEnginePath(gpu_id, "x100_big_locate");
        if (!loadEngine(enginePath)) {
            sample::gLogError << "Failed to load engine: " << enginePath << std::endl;
        }	
	}
	bool infer(cv::Mat& image, std::vector<cv::Rect>& out)
	{
		org_w = image.cols;
		org_h = image.rows;
		std::vector<cv::Mat> uImgs;
		uImgs.push_back(image);
		std::vector<float> inputData = processInput(uImgs);
        
        // 调用基类执行推理
        if (!doInference({{"input.1", inputData}})) return false;

        // 获取输出
        float* ptr_pool_cls = mHostOutputs["330"].data();
		float* ptr_pred_cls = mHostOutputs["329"].data();
		float* ptr_pred_size = mHostOutputs["325"].data();
		float* ptr_pred_offset = mHostOutputs["328"].data();

        postprocess(ptr_pool_cls, ptr_pred_cls, ptr_pred_size, ptr_pred_offset, out);
        return true;
	}
private:
	int org_w = 2448;
	int org_h = 2048;
	std::vector<float> processInput(std::vector<cv::Mat> srclist)
	{
		int len = BATCH * INPUTC * INPUTW * INPUTH;
        std::vector<float> chw(len);
		float* data = chw.data();

		for (int m = 0; m < BATCH; m++)
		{	
			cv::Mat image;
			cv::resize(srclist[m], image, cv::Size(306, 256), 0.0, 0.0, cv::INTER_NEAREST);
			cv::Mat image_(INPUTH, INPUTW, CV_8UC3, cv::Scalar(255, 255, 255));
			image_(cv::Rect(0, 0, image.cols, image.rows)) = image + 0;
			image_.convertTo(image_, CV_32FC3);
			image_ = image_ / 255.0;

			std::vector<cv::Mat> channels(INPUTC);
			for (int c = 0; c < INPUTC; ++c)
			{
				// 每个通道指向 chw 向量中对应的平面起始位置
				channels[c] = cv::Mat(INPUTH, INPUTW, CV_32FC1, data + m * INPUTC * INPUTH * INPUTW + c * INPUTH * INPUTW);
			}
			// 将 HWC 的 image 拆分并直接拷贝到 channels 指向的 chw 内存中
			cv::split(image_, channels);
		}

		return chw;
	}
	//后处理
	void postprocess(float* pool_cls, float* pred_cls, float* pred_size, float* pred_offset, std::vector<cv::Rect>& out) 
	{
		float* ptr_cls = pred_cls;
		float* ptr_pool_cls = pool_cls;

		const int img_width = org_w;
		const int img_height = org_h;

		for (int i = 0; i < BATCH; i++) {
			// 用于 NMS
			std::vector<cv::Rect> localBoxes;
			std::vector<float> localConfidences;
			for (int j = 0; j < 64 * 80; j++) {
				// cout << *ptr_cls << " " << *ptr_pool_cls << endl;
				if (*ptr_cls == *ptr_pool_cls && (*ptr_cls) >= 0.3)
				{
					
					int center_x = j % 80;
					int center_y = j / 80;
					
					int x_min = (center_x * 4 + pred_offset[(center_x)+(80 * center_y) + (80 * 64 * 0) + (80 * 64 * 2 * i)]
						- pred_size[(center_x)+(80 * center_y) + (80 * 64 * 0) + (80 * 64 * 2 * i)] * 80 / 2) / 306 * img_width;

					int y_min = (center_y * 4 + pred_offset[(center_x)+(80 * center_y) + (80 * 64 * 1) + (80 * 64 * 2 * i)]
						- pred_size[(center_x)+(80 * center_y) + (80 * 64 * 1) + (80 * 64 * 2 * i)] * 64 / 2) / 256 * img_height;

					int x_max = (center_x * 4 + pred_offset[(center_x)+(80 * center_y) + (80 * 64 * 0) + (80 * 64 * 2 * i)]
						+ pred_size[(center_x)+(80 * center_y) + (80 * 64 * 0) + (80 * 64 * 2 * i)] * 80 / 2) / 306 * img_width;

					int y_max = (center_y * 4 + pred_offset[(center_x)+(80 * center_y) + (80 * 64 * 1) + (80 * 64 * 2 * i)]
						+ pred_size[(center_x)+(80 * center_y) + (80 * 64 * 1) + (80 * 64 * 2 * i)] * 64 / 2) / 256 * img_height;


					cv::Rect box = cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min);
					localBoxes.push_back(box);
					localConfidences.push_back(*ptr_cls);

				}
				ptr_cls++;
				ptr_pool_cls++;
			}
			std::vector<int> nmsIndices;
			cv::dnn::NMSBoxes(localBoxes, localConfidences, 0.6f, 0.3f, nmsIndices);
			for (int ii = 0; ii < nmsIndices.size(); ii++) 
			{
				int idx_ = nmsIndices[ii];
				cv::Rect rect(0, 0, img_width, img_height);
				cv::Rect cell_rect = localBoxes[idx_] & rect;
				out.push_back(cell_rect);
			}	
		}
	}
};
