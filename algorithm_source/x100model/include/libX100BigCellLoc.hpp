#pragma once
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

typedef std::pair<float, float> ratio_;
typedef std::tuple<float, int, int> class_conf_idx_label;


class X100BigLocateOnnx
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	vector<vector<float> > anchor_boxes;
	X100BigLocateOnnx(int gpu_id) :sortThreadValue(0.5)
	{
		mParams.inputTensorNames.push_back("input.1");
		mParams.batchSize = 1;
	
		mParams.outputTensorNames.push_back("330");//292
		mParams.outputTensorNames.push_back("329");//292
		mParams.outputTensorNames.push_back("325");//287
		mParams.outputTensorNames.push_back("328");//290
		mParams.dlaCore = 1;
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
			engine = "engines/2080/x100_big_locate.trt";
		else if(index_3080 != string::npos)
			engine = "engines/3080/x100_big_locate.trt";
		else if(index_4070 != string::npos)
			engine = "engines/4070/x100_big_locate.trt";
		else if(index_4090 != string::npos)
			engine = "engines/4070/x100_big_locate.trt";
		else
			std::cout << "cannot find correct trt" << std::endl;

		// std::string engine = "engines/x100_big_locate.trt";
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
		
	}
	~X100BigLocateOnnx()
	{}
	bool infer(cv::Mat& image, vector<cv::Rect>& out)
	{
		samplesCommon::BufferManager buffers(mEngine);
    	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
		if (!context)
		{
			return false;
		}

		// Read the input data into the managed buffers
		assert(mParams.inputTensorNames.size() == 1);
		if (!processInput(buffers, vector<cv::Mat>{image}))
		{
			return false;
		}

		buffers.copyInputToDevice();



		/*double time = static_cast<double>(cv::getTickCount());*/

		bool status = context->executeV2(buffers.getDeviceBindings().data());
		/*time1 = ((double)cv::getTickCount() - time1) / cv::getTickFrequency();
		sample::gLogInfo << "copy + executeV2	" << time1 << std::endl;*/
		if (!status)
		{
			return false;
		}

		// Memcpy from device output buffers to host output buffers
		buffers.copyOutputToHost();

		float* pred_cls = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));
		float* pool_cls = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
		float* pred_size = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[2]));
		float* pred_offset = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[3]));

		float* ptr_cls = pred_cls;
		float* ptr_pool_cls = pool_cls;

		const int img_width = image.cols;
		const int img_height = image.rows;
		int batchSize = 1;
		// 每张图片单独处理
		for (int i = 0; i < batchSize; i++) {
			// 用于 NMS
			vector<Rect> localBoxes;
			vector<float> localConfidences;
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


					Rect box = Rect(x_min, y_min, x_max - x_min, y_max - y_min);
					localBoxes.push_back(box);
					localConfidences.push_back(*ptr_cls);

				}
				ptr_cls++;
				ptr_pool_cls++;
			}
			std::cout << "localBoxes.size() --->>> " << localBoxes.size() << std::endl;
			vector<int> nmsIndices;
			cv::dnn::NMSBoxes(localBoxes, localConfidences, 0.6f, 0.3f, nmsIndices);

			//vector<Rect> out_boxes;  // 本张图片最终输出结果
			//vector<int> scores;    // 上面那些框的预测值
			std::cout << "nmsIndices.size() --->>> " << nmsIndices.size() << std::endl;
			for (int ii = 0; ii < nmsIndices.size(); ii++) 
			{
				int idx_ = nmsIndices[ii];
				cv::Rect rect(0, 0, img_width, img_height);
				cv::Rect cell_rect = localBoxes[idx_] & rect;
				out.push_back(cell_rect);
			}
	
		}
		return true;
	}
private:
	samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.
	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
	int mNumber{ 0 };             //!< The number to classify
	// std::shared_ptr<nvinfer1::ICudaEngine> mEngine; !< The TensorRT engine used to run the network
	std::shared_ptr<nvinfer1::IRuntime> mRuntime; 
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
	float sortThreadValue; // 用于筛选框选结果是否有效的判断值
	bool processInput(const samplesCommon::BufferManager& buffers, vector<cv::Mat> srclist)
	{
		const int inputC = 3;
		const int inputH = 256;
		const int inputW = 320;
		const int batchSize = 1;

	
		cv::Scalar mean_(0.597, 0.519, 0.521);
		cv::Scalar std_(0.311, 0.329, 0.327);
		float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
		for (int m = 0; m < batchSize; m++)
		{
			
			cv::Mat image;
			cv::resize(srclist[m], image, cv::Size(306, 256), 0.0, 0.0, cv::INTER_NEAREST);
			cv::Mat image_(inputH, inputW, CV_8UC3, cv::Scalar(255, 255, 255));
			image_(cv::Rect(0, 0, image.cols, image.rows)) = image + 0;
			image_.convertTo(image_, CV_32FC3);


			image_ = image_ / 255;

			for (int i = 0; i < inputH; i++) {
				float* data = image_.ptr<float>(i);
				for (int j = 0; j < inputW; j++) {
					for (int k = 0; k < inputC; k++) {
						hostDataBuffer[m*inputH*inputW*inputC + k * inputW*inputH + i * inputW + j] = data[j*inputC + k];
					}
				}
			}
		}

		return true;
	}
};
