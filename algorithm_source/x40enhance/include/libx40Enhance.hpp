
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "opencv2/cudawarping.hpp"
#include <type_traits>
#include <cuda_runtime.h>
using namespace cv;
using namespace std;

class EnhanceOnnx
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:

	int device;
	EnhanceOnnx(int gpu_id)
	{
		mParams.inputTensorNames.push_back("x");
		mParams.batchSize = 1;
		mParams.outputTensorNames.push_back("1448");
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
			engine = "engines/2080/x40_enhance.trt";
		else if(index_3080 != string::npos)
			engine = "engines/3080/x40_enhance.trt";
		else if(index_4070 != string::npos)
			engine = "engines/4070/x40_enhance.trt";
		else if(index_4090 != string::npos)
			engine = "engines/4070/x40_enhance.trt";
		else
			std::cout << "cannot find correct trt" << std::endl;

		// std::string engine = "engines/x40_enhance.trt";
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
	~EnhanceOnnx()
	{};

	bool infer(cv::Mat uImg, cv::Mat& uOutImg)
	{
		samplesCommon::BufferManager buffers(mEngine);
    	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
		if (!context)
		{
			return false;
		}
		// Read the input data into the managed buffers
		assert(mParams.inputTensorNames.size() == 1);

		//double time1 = static_cast<double>(cv::getTickCount());
		
		if (!processInput(buffers, uImg))
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

		const int batchSize = 1;
		const int inputC = 3;
		float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
	
		const int inputH = 2048;
		const int inputW = 2448;
		const int outputSize = inputH * inputW;
	
		for (int b = 0; b < batchSize; b++)
		{
			cv::Mat result = cv::Mat::zeros(cv::Size(inputW, inputH), CV_32FC3);
			std::vector<cv::Mat> chw{
			cv::Mat(inputH, inputW, CV_32F, &output[2 * outputSize]),
			cv::Mat(inputH, inputW, CV_32F, &output[1 * outputSize]),
			cv::Mat(inputH, inputW, CV_32F, &output[0 * outputSize])
			};
			cv::merge(chw, result);
			result.convertTo(result, CV_8UC3, 255);
			uOutImg = result + 0;
		}	
		return true;
	}


private:
	samplesCommon::OnnxSampleParams mParams; 
	std::shared_ptr<nvinfer1::IRuntime> mRuntime; 
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
	int mNumber{ 0 };             //!< The number to classify

	bool processInput(const samplesCommon::BufferManager& buffers, cv::Mat uImg)
	{
		const int inputC = 3;
		const int inputH = 1024;
		const int inputW = 1224;
		float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
		cv::resize(uImg, uImg, cv::Size(), .5, .5);
		cv::Mat rgb;
		cv::cvtColor(uImg, rgb, cv::COLOR_BGR2RGB);
		rgb.convertTo(rgb, CV_32FC3);
		uImg = rgb / 255;

		for (int i = 0; i < inputH; i++) {
				float* data = uImg.ptr<float>(i);

				for (int j = 0; j < inputW; j++) {
					for (int k = 0; k < inputC; k++) {
						hostDataBuffer[0*inputH*inputW*inputC + k * inputW*inputH + i * inputW + j] = data[j*inputC + k];
					}
				}
			}
		return true;
	}
};

