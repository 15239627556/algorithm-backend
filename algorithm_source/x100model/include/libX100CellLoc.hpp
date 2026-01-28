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

class X100HaveLocateOnnx: public PublicTRT
{
public:
	static inline const int INPUTC = 3;
	static inline const int INPUTH = 384;
	static inline const int INPUTW = 512;
	static inline const int BATCH = 1;
	static inline const int CLASSNUM = 1;
	static inline const float THRESHOLD = 0.3f;
	static inline const float CONFTHRESHOLD = 0.3f;
	static inline const float NMSTHRESHOLD = 0.2f;
	static inline const cv::Scalar mean_ = cv::Scalar(0.485, 0.456, 0.406);
	static inline const cv::Scalar std_ = cv::Scalar(0.229, 0.224, 0.225);
	
	X100HaveLocateOnnx(int gpu_id)
	{
		initTRT();
        std::string enginePath = selectEnginePath(gpu_id, "x100_have_locate");
        if (!loadEngine(enginePath)) {
            sample::gLogError << "Failed to load engine: " << enginePath << std::endl;
        }	

		float anchor_scale = 2;  
		std::vector<float> pyramid_levels{ 4, 5, 6 }; 
		std::vector<float> strides;
		for (size_t i = 0; i < pyramid_levels.size(); i++) {
			int pow_ = pow(2, pyramid_levels[i]);
			strides.push_back(pow_);
		}
		std::vector<float> scales{ 1, 1.4142135623730951f }; 
		std::vector<ratio_> ratios{ std::make_pair(1.0, 1.0), std::make_pair(1.2, 0.8), std::make_pair(0.8, 1.2) };
		for (size_t i = 0; i < strides.size(); i++) {
			int index_m = ceil((INPUTW - strides[i] / 2) / strides[i]);
			int index_n = ceil((INPUTH - strides[i] / 2) / strides[i]);
			std::vector<std::vector<int> > xv(index_n, std::vector<int>(index_m));
			std::vector<std::vector<int> > yv(index_n, std::vector<int>(index_m));
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
        if (!doInference({{"data", inputData}})) return false;

        // 获取输出
        float* regression = mHostOutputs["1228"].data();
		float* classification = mHostOutputs["1392"].data();

        postprocess(regression, classification, out);
        return true;
	}

private:
	std::vector<std::vector<float> > anchor_boxes;
	typedef std::tuple<float, int, int> class_conf_idx_label;
	typedef std::pair<float, float> ratio_;
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
	void postprocess(float* regression, float* classification, std::vector<cv::Rect>& out) 
	{
		auto max_index = [](const float* start, const float* end) -> int {
			float max_val = start[0];
			int max_pos = 0;
			for (size_t i = 1; start + i < end; ++i) {
				if (start[i] > max_val) {
					max_val = start[i];
					max_pos = int(i);
				}
			}

			return max_pos;
		};

		std::vector<class_conf_idx_label> class_conf_idx_labels;
		

		for (size_t i = 0; i < anchor_boxes.size(); i++) {
			auto max_idx = max_index(classification + i * CLASSNUM, classification + (i + 1)*CLASSNUM);
			if (classification[i*CLASSNUM + max_idx] > THRESHOLD)
				class_conf_idx_labels.push_back(std::make_tuple(classification[i*CLASSNUM + max_idx], i, max_idx));
		}

		for (size_t i = 0; i < CLASSNUM; i++) {
			std::vector<cv::Rect> localBoxes;
			std::vector<float> localConfidences;
			for (size_t j = 0; j < class_conf_idx_labels.size(); j++) {
				if (std::get<2>(class_conf_idx_labels[j]) == i) {

					int idx = std::get<1>(class_conf_idx_labels[j]);
					float conf = std::get<0>(class_conf_idx_labels[j]);

					float y_centers_a = (anchor_boxes[idx][0] + anchor_boxes[idx][2]) / 2;
					float x_centers_a = (anchor_boxes[idx][1] + anchor_boxes[idx][3]) / 2;
					float ha = anchor_boxes[idx][2] - anchor_boxes[idx][0];
					float wa = anchor_boxes[idx][3] - anchor_boxes[idx][1];

					float w = exp(regression[idx * 4 + 3]) * wa;
					float h = exp(regression[idx * 4 + 2]) * ha;

					float y_centers = regression[idx * 4 + 0] * ha + y_centers_a;
					float x_centers = regression[idx * 4 + 1] * ha + x_centers_a;
					float zero = 0;

					cv::Rect box = cv::Rect(int(std::max(x_centers - w / 2, zero)), int(std::max(y_centers - h / 2, zero)),
						int(std::min(w, float(INPUTH - 1))), int(std::min(h, float(INPUTW - 1))));
					localBoxes.push_back(box);
					localConfidences.push_back(conf);
				}
			}
			std::vector<int> nmsIndices;
			cv::dnn::NMSBoxes(localBoxes, localConfidences, CONFTHRESHOLD, NMSTHRESHOLD, nmsIndices);

			for (size_t idx = 0; idx < nmsIndices.size(); idx++)
			{
				size_t idx_ = nmsIndices[idx];
				{
					cv::Rect rctOut = localBoxes.at(idx_);
					int left = rctOut.x * org_w / INPUTW;
					int top = rctOut.y * org_h / INPUTH;
					int right = (rctOut.x + rctOut.width) * org_w / INPUTW;
					int bottom = (rctOut.y + rctOut.height) * org_h / INPUTH;
					out.push_back(cv::Rect(left, top, right - left, bottom - top));
				}
			}
		}
	}
};



