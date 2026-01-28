
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


typedef std::pair<float, float> ratio_;
typedef std::tuple<float, int, int> class_conf_idx_label;

class X100RBCLocateOnnx: public PublicTRT
{
	static inline const int BATCH = 1;
	static inline const int INPUTC = 3;
	static inline const int INPUTH = 384;
	static inline const int INPUTW = 512;
	static inline const float THRESHOLD = 0.3;
	static inline const int OUTW = 128;
	static inline const int OUTH = 96;
	static inline const cv::Scalar mean_ = cv::Scalar(123.675, 116.28, 103.53);
	static inline const cv::Scalar std_ = cv::Scalar(58.395, 57.12, 57.375);
public:
	X100RBCLocateOnnx(int gpu_id)
	{
		initTRT();
        std::string enginePath = selectEnginePath(gpu_id, "x100_RBC_locate");
        if (!loadEngine(enginePath)) {
            sample::gLogError << "Failed to load engine: " << enginePath << std::endl;
        }	
		
		float anchor_scale = 2;  // constant !!!!!
		std::vector<float> pyramid_levels{ 4, 5, 6 }; // layers
		std::vector<float> strides;
		for (size_t i = 0; i < pyramid_levels.size(); i++) {
			int pow_ = pow(2, pyramid_levels[i]);
			strides.push_back(pow_);
		}
		std::vector<float> scales{ 1, 1.4142135623730951 }; // (2^0, 2^(1/3), 2^(2/3))
		std::vector<ratio_> ratios{ std::make_pair(1.0, 1.0), std::make_pair(1.2, 0.8), std::make_pair(0.8, 1.2) };
		for (size_t i = 0; i < strides.size(); i++) {
			int index_m = ceil((INPUTW - strides[i] / 2) / strides[i]);
			int index_n = ceil((INPUTH - strides[i] / 2) / strides[i]);
			std::vector<std::vector<int> > xv(index_n, std::vector<int>(index_m));
			std::vector<std::vector<int> > yv(index_n, std::vector<int>(index_m));
			//float xv[index_n*index_m] = {0}, yv[index_n*index_m] = {0};
			for (size_t n = 0; n < index_n; n++) {
				for (size_t m = 0; m < index_m; m++) {
					for (size_t j = 0; j < scales.size(); j++) {
						for (size_t k = 0; k < ratios.size(); k++) {
							float base_anchor_size = anchor_scale * strides[i] * scales[j];
							float anchor_size_x_2 = base_anchor_size * ratios[k].first / 2;
							float anchor_size_y_2 = base_anchor_size * ratios[k].second / 2;
							xv[n][m] = strides[i] / 2 + m * strides[i];
							yv[n][m] = strides[i] / 2 + n * strides[i];
							std::vector<float> box{ yv[n][m] - anchor_size_y_2, xv[n][m] - anchor_size_x_2,
								yv[n][m] + anchor_size_y_2, xv[n][m] + anchor_size_x_2 };
							anchor_boxes.push_back(box);

						}
					}
				}
			}
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
        if (!doInference({{"input", inputData}})) return false;

        // 获取输出
        float* cls = mHostOutputs["256"].data();
		float* clsMax = mHostOutputs["263"].data();
		float* wh = mHostOutputs["259"].data();
		float* offset = mHostOutputs["262"].data();

        postprocess(cls, clsMax, wh, offset, out);
        return true;
	}

private:
	std::vector<std::vector<float> > anchor_boxes;
	int org_w = 2048;
	int org_h = 1536;

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
	void postprocess(float* cls, float* clsMax, float* wh, float* offset, std::vector<cv::Rect>& out) 
	{
		for (int i = 0; i < BATCH; i++) {
			// vector<itmRedCellLocInfo> instances{};
			int clsStartPos = i * OUTW * OUTH;
			cv::Mat clsMat = cv::Mat(OUTH, OUTW, CV_32FC1, cls + clsStartPos);
			cv::Mat clsMaxMat = cv::Mat(OUTH, OUTW, CV_32FC1, clsMax + i);
			cv::Mat mask = (clsMat > THRESHOLD) & (abs(clsMat - clsMaxMat) < 1e-6);
			std::vector<cv::Point> locations;
			cv::findNonZero(mask, locations);

			cv::Mat xOffsetMat = cv::Mat(OUTH, OUTW, CV_32FC1,
				offset + clsStartPos * 2);
			cv::Mat yOffsetMat = cv::Mat(OUTH, OUTW, CV_32FC1,
				offset + clsStartPos * 2 + OUTW * OUTH);
			cv::Mat wMat = cv::Mat(OUTH, OUTW, CV_32FC1, wh + clsStartPos * 2);
			cv::Mat hMat = cv::Mat(OUTH, OUTW, CV_32FC1,
				wh + clsStartPos * 2 + OUTW * OUTH);

			for (cv::Point loc : locations) {
				int x = loc.x;
				int y = loc.y;
				float centerX = ((float)x + xOffsetMat.at<float>(loc)) / OUTW * org_w;
				float centerY = ((float)y + yOffsetMat.at<float>(loc)) / OUTH * org_h;
				float width = wMat.at<float>(loc) / OUTW * org_w;
				float height = hMat.at<float>(loc) / OUTH * org_h;
				cv::Rect cellRect = cv::Rect((int)(centerX - width / 2), (int)(centerY - height / 2), (int)width, (int)height);
				out.push_back(cellRect & cv::Rect(0, 0, org_w, org_h));
			}
		}
	}
};

