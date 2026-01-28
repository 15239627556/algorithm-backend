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

struct PicCellAnalysisResult{
	cv::Mat uOutPic;
	std::vector<cv::Point> cellCenterPoints;
};

class CellAnalysisOnnx : public PublicTRT {
public:
	// 将静态常量定义在类内部
    static inline const int BATCH  = 4;
    static inline const int INPUTH = 512;
    static inline const int INPUTW = 640;
    static inline const int INPUTC = 4;

    CellAnalysisOnnx(int gpu_id) {
        initTRT();
        std::string enginePath = selectEnginePath(gpu_id, "x40_cellAnalysis");
        if (!loadEngine(enginePath)) {
            sample::gLogError << "Failed to load engine: " << enginePath << std::endl;
        }
    }

    bool infer(std::vector<cv::Mat> uImgs, std::vector<PicCellAnalysisResult>& out) {
        // 预处理
        std::vector<float> inputData = processInput(uImgs);
        std::cout << "processInput end " << std::endl;
        // 组织输入 (即使只有一个输入也用 Map)
        std::unordered_map<std::string, std::vector<float>> inputs = {{"input", inputData}};

        // 推理
        if (!doInference(inputs)) return false;

		std::cout << "doInference end " << std::endl;

        // 直接从基类的 mHostOutputs 中取数据，名字对应 ONNX 的 Output Name
        float* pred_cls = mHostOutputs["center"].data();
        float* pool_cls = mHostOutputs["center_pool"].data();
		float* pred_size = mHostOutputs["size"].data();
        float* mask     = mHostOutputs["sem_seg"].data();

        postprocess(pred_cls, pool_cls, pred_size, mask, out);
		std::cout << "postprocess end " << std::endl;
        return true;
    }

private:
	//前处理
	std::vector<float>  processInput(const std::vector<cv::Mat>& uImgs)
	{
		if (uImgs.size() < BATCH) {
        throw std::runtime_error("Input images count is less than BATCH size!");
    	}
		int len = BATCH * INPUTC * INPUTW * INPUTH;
        std::vector<float> chw(len);
		float* data = chw.data();
		for (int m = 0; m < BATCH; m++)
		{
			cv::Mat HSV;
			cv::cvtColor(uImgs[m], HSV, cv::COLOR_BGR2HSV);
			std::vector<cv::Mat> channels;
			cv::split(HSV, channels);

			std::vector<cv::Mat> channels_;
			cv::split(uImgs[m], channels_);
			channels_.push_back(channels.at(1));
			cv::Mat image;
			cv::merge(channels_, image);
			image.convertTo(image, CV_32FC4);
			image = image / 255.0;

			std::vector<cv::Mat> channels2(INPUTC);
			for (int c = 0; c < INPUTC; ++c)
			{
				// 每个通道指向 chw 向量中对应的平面起始位置
				channels2[c] = cv::Mat(INPUTH, INPUTW, CV_32FC1, data + m * INPUTC * INPUTH * INPUTW + c * INPUTH * INPUTW);
			}
			// 将 HWC 的 image 拆分并直接拷贝到 channels2 指向的 chw 内存中
			cv::split(image, channels2);
		}		
		return chw;
	}
	//后处理
	void postprocess(float* pred_cls, float* pool_cls, float* pred_size, float* mask, std::vector<PicCellAnalysisResult>& uOutImg)
	{
		float* ptr_cls = pred_cls;
		float* ptr_pool_cls = pool_cls;
		for (int b = 0; b < BATCH; b++)
		{
			PicCellAnalysisResult uOutImg_singel;
			cv::Mat bgMask = cv::Mat(INPUTH, INPUTW, CV_32FC1, mask + b * (3 * INPUTH * INPUTW));
			cv::Mat whiteMask = cv::Mat(INPUTH, INPUTW, CV_32FC1, mask + b * (3 * INPUTH * INPUTW) + INPUTH * INPUTW);
			cv::Mat redMask = cv::Mat(INPUTH, INPUTW, CV_32FC1, mask + b * (3 * INPUTH * INPUTW) + 2 * INPUTH * INPUTW);

			cv::Mat bgr;
			std::vector<cv::Mat> channels = { whiteMask , bgMask , redMask };
			cv::merge(channels, bgr);
			bgr = bgr * 255;
			bgr.convertTo(bgr, CV_8UC3);

			uOutImg_singel.uOutPic = bgr(cv::Rect(0, 0, 612, 512));

			std::vector<cv::Rect> localBoxes;
			std::vector<float> localConfidences;
			for (int j = 0; j < INPUTH * INPUTW; j++) 
			{
				
				if (*ptr_cls == *ptr_pool_cls && (*ptr_cls) >= 0.2) {

					int center_x = j % 640;
					int center_y = j / 640;

					if (whiteMask.at<float>(center_y, center_x) < .3)
					{
						ptr_cls++;
						ptr_pool_cls++;
						continue;
					}

					int x_min = (center_x - 25 / 2);
											
					int y_min = (center_y - 25 / 2);
											
					int x_max = (center_x + 25 / 2);
											
					int y_max = (center_y + 25 / 2);


					cv::Rect box = cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min);
					localBoxes.push_back(box);
					localConfidences.push_back(*ptr_cls);

				}
				ptr_cls++;
				ptr_pool_cls++;

			}		
			std::vector<cv::Rect> out_boxes;  
			std::vector<double> scores;  
			for (size_t idx = 0; idx < localBoxes.size(); idx++) {
				
				out_boxes.push_back(localBoxes[idx]/* + cv::Point(28,0)*/);
				scores.push_back(localConfidences[idx]);
			}
			
			for (size_t ki = 0; ki < out_boxes.size(); ki++)
			{
				cv::Rect rect(0, 0, 612, 512);
				out_boxes[ki] = out_boxes[ki] & rect;
			}
			
			for (size_t k = 0; k < out_boxes.size(); k++)
			{
				int c_x = out_boxes[k].x + out_boxes[k].width / 2;
				int c_y = out_boxes[k].y + out_boxes[k].height / 2;
				if (c_x >= 0 && c_x <= 612 && c_y >= 0 && c_y <= 512)
					uOutImg_singel.cellCenterPoints.push_back(cv::Point(c_x, c_y));
			}			
			uOutImg.push_back(uOutImg_singel);
		}	
		return;
	}
};
