#pragma once
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;


typedef std::pair<float, float> ratio_;
typedef std::tuple<float, int, int> class_conf_idx_label;

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

class X100HaveLocateOnnx
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	X100HaveLocateOnnx(int gpu_id) :sortThreadValue(0.5)
	{
		mParams.inputTensorNames.push_back("data");
		mParams.batchSize = 1;
		mParams.outputTensorNames.push_back("1228");
		mParams.outputTensorNames.push_back("1392");
		mParams.dlaCore = -1;
		mParams.int8 = false;
		mParams.fp16 = false;

		initLibNvInferPlugins(nullptr, "");
		// cudaDeviceProp deviceProp;
		// cudaGetDeviceProperties(&deviceProp, gpu_id);
		// string diviceName = deviceProp.name;
		string diviceName = GPU_NAMES[gpu_id];
		size_t index_2080 = diviceName.find("2080");
		size_t index_3080 = diviceName.find("3080");
		size_t index_4070 = diviceName.find("4070");
		size_t index_4090 = diviceName.find("4090");
		std::string engine = "";
		if(index_2080 != string::npos)
			engine = "engines/2080/x100_have_locate.trt";
		else if(index_3080 != string::npos)
			engine = "engines/3080/x100_have_locate.trt";
		else if(index_4070 != string::npos)
			engine = "engines/4070/x100_have_locate.trt";
		else if(index_4090 != string::npos)
			engine = "engines/4070/x100_have_locate.trt";
		else
			std::cout << "cannot find correct trt" << std::endl;

		// std::string engine = "engines/x100_have_locate.trt";
		std::ifstream engineFile(engine, std::ios::binary);
		if (!engineFile)
		{
			sample::gLogInfo << "Error opening engine file: " << engine << std::endl;
			return ;
		}
		engineFile.seekg(0, engineFile.end);
		long int fsize = engineFile.tellg();
		engineFile.seekg(0, engineFile.beg);

		std::vector<char> engineData(fsize);
		engineFile.read(engineData.data(), fsize);
		if (!engineFile)
		{
			sample::gLogInfo << "Error loading engine file: " << engine << std::endl;
			return ;
		}
		sample::gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kERROR);  // 设置日志级别
		mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
		mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(engineData.data(), fsize, nullptr));

		float anchor_scale = 2;  

		vector<float> pyramid_levels{ 4, 5, 6 }; 
		vector<float> strides;
		for (size_t i = 0; i < pyramid_levels.size(); i++) {
			int pow_ = pow(2, pyramid_levels[i]);
			strides.push_back(pow_);
		}
		vector<float> scales{ 1, 1.4142135623730951f }; 
		vector<ratio_> ratios{ make_pair(1.0, 1.0), make_pair(1.2, 0.8), make_pair(0.8, 1.2) };
		for (size_t i = 0; i < strides.size(); i++) {
			int index_m = ceil((image_shape[1] - strides[i] / 2) / strides[i]);
			int index_n = ceil((image_shape[0] - strides[i] / 2) / strides[i]);
			vector<vector<int> > xv(index_n, vector<int>(index_m));
			vector<vector<int> > yv(index_n, vector<int>(index_m));
			for (size_t n = 0; n < index_n; n++) {
				for (size_t m = 0; m < index_m; m++) {
					for (size_t j = 0; j < scales.size(); j++) {
						for (size_t k = 0; k < ratios.size(); k++) {
							float base_anchor_size = anchor_scale * strides[i] * scales[j];
							float anchor_size_x_2 = base_anchor_size * ratios[k].first / 2;
							float anchor_size_y_2 = base_anchor_size * ratios[k].second / 2;
							xv[n][m] = strides[i] / 2 + m * strides[i];
							yv[n][m] = strides[i] / 2 + n * strides[i];
							vector<float> box{ yv[n][m] - anchor_size_y_2, xv[n][m] - anchor_size_x_2,
								yv[n][m] + anchor_size_y_2, xv[n][m] + anchor_size_x_2 };
							anchor_boxes.push_back(box);

						}
					}
				}
			}
		}
	}

	~X100HaveLocateOnnx()
	{}

	bool infer(cv::Mat& image, vector<cv::Rect>& out)
	{
		out.clear();
		samplesCommon::BufferManager buffers(mEngine);
    	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
		if (!context)
		{
			return false;
		}

		assert(mParams.inputTensorNames.size() == 1);
		if (!processInput(buffers, image))
		{
			return false;
		}
		buffers.copyInputToDevice();

		bool status = context->executeV2(buffers.getDeviceBindings().data());
		if (!status)
		{
			return false;
		}

		buffers.copyOutputToHost();

		float* regression = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
		float* classification = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));

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

		vector<class_conf_idx_label> class_conf_idx_labels;
		int class_num = 1;
		float threshold = 0.3f, confThreshold = 0.3f, nmsThreshold = 0.2f;

		for (size_t i = 0; i < anchor_boxes.size(); i++) {
			auto max_idx = max_index(classification + i * class_num, classification + (i + 1)*class_num);
			if (classification[i*class_num + max_idx] > threshold)
				class_conf_idx_labels.push_back(make_tuple(classification[i*class_num + max_idx], i, max_idx));
		}

		for (size_t i = 0; i < class_num; i++) {
			vector<Rect> localBoxes;
			vector<float> localConfidences;
			for (size_t j = 0; j < class_conf_idx_labels.size(); j++) {
				if (get<2>(class_conf_idx_labels[j]) == i) {

					int idx = get<1>(class_conf_idx_labels[j]);
					float conf = get<0>(class_conf_idx_labels[j]);

					float y_centers_a = (anchor_boxes[idx][0] + anchor_boxes[idx][2]) / 2;
					float x_centers_a = (anchor_boxes[idx][1] + anchor_boxes[idx][3]) / 2;
					float ha = anchor_boxes[idx][2] - anchor_boxes[idx][0];
					float wa = anchor_boxes[idx][3] - anchor_boxes[idx][1];

					float w = exp(regression[idx * 4 + 3]) * wa;
					float h = exp(regression[idx * 4 + 2]) * ha;

					float y_centers = regression[idx * 4 + 0] * ha + y_centers_a;
					float x_centers = regression[idx * 4 + 1] * ha + x_centers_a;
					float zero = 0;

					Rect box = Rect(int(std::max(x_centers - w / 2, zero)), int(std::max(y_centers - h / 2, zero)),
						int(std::min(w, image_shape[0] - 1)), int(std::min(h, image_shape[1] - 1)));
					localBoxes.push_back(box);
					localConfidences.push_back(conf);
				}
			}
			vector<int> nmsIndices;
			cv::dnn::NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);

			for (size_t idx = 0; idx < nmsIndices.size(); idx++)
			{
				size_t idx_ = nmsIndices[idx];
				{
					Rect rctOut = localBoxes.at(idx_);
					int left = rctOut.x * image.cols / 512;
					int top = rctOut.y * image.rows / 384;
					int right = (rctOut.x + rctOut.width) * image.cols / 512;
					int bottom = (rctOut.y + rctOut.height) * image.rows / 384;
					out.push_back(Rect(left, top, right - left, bottom - top));
				}
			}
		}

		return true;
	}

private:
	samplesCommon::SampleParams mParams; //!< The parameters for the sample.
	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
	int mNumber{ 0 };             //!< The number to classify
	std::shared_ptr<nvinfer1::IRuntime> mRuntime; 
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
	float sortThreadValue;
	vector<vector<float> > anchor_boxes;
	vector<float> image_shape{ 384,512 };
	
	bool processInput(const samplesCommon::BufferManager& buffers, cv::Mat& src)
	{
		const int inputC = 3;
		const int inputH = 384;
		const int inputW = 512;
		const int batchSize = 1;

		cv::Scalar mean_(0.485, 0.456, 0.406);
		cv::Scalar std_(0.229, 0.224, 0.225);
		cv::Mat image;
		cv::resize(src, image, cv::Size(inputW, inputH));
		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

		image.convertTo(image, CV_32FC3);
		image = image / 255;
		cv::subtract(image, mean_, image);
		cv::divide(image, std_, image);

	
		float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

		for (int i = 0; i < inputH; i++) {
			float* data = image.ptr<float>(i);
			for (int j = 0; j < inputW; j++) {
				for (int k = 0; k < inputC; k++) {
					hostDataBuffer[k*inputW*inputH + i * inputW + j] = data[j*inputC + k];
				}
			}
		}

		return true;
	}
};



