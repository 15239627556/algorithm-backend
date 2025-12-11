#pragma once
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
#include "common.hpp"
#include <cuda_runtime.h>
using namespace std;

class X100BigClassifyOnnx
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    X100BigClassifyOnnx(int gpu_id)       : mEngine(nullptr)
    {
	    mParams.inputTensorNames.push_back("input.1");
	    mParams.batchSize = 1;
	    mParams.outputTensorNames.push_back("226");
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
			engine = "engines/2080/x100_big_classify.trt";
		else if(index_3080 != string::npos)
			engine = "engines/3080/x100_big_classify.trt";
		else if(index_4070 != string::npos)
			engine = "engines/4070/x100_big_classify.trt";
		else if(index_4090 != string::npos)
			engine = "engines/4070/x100_big_classify.trt";
		else
			std::cout << "cannot find correct trt" << std::endl;

		// std::string engine = "engines/x100_big_classify.trt";
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
	~X100BigClassifyOnnx()
	{
	}
    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(cv::Mat& image, std::vector<itmCellRcgz_x100>& outList)
    {
		samplesCommon::BufferManager buffers(mEngine);
    	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	   
		if (!context)
		{
			return false;
		}
	    // Read the input data into the managed buffers
	    assert(mParams.inputTensorNames.size() == 1);
	    if (!processInput(buffers, image))
	    {
	        return false;
	    }
		
	    // Memcpy from host input buffers to device input buffers
		
		buffers.copyInputToDevice();

	    bool status = context->executeV2(buffers.getDeviceBindings().data());

	    if (!status)
	    {
	        return false;
	    }
	    // Memcpy from device output buffers to host output buffers
	    buffers.copyOutputToHost();
		
	    const int outputSize = 14; // mOutputDims.d[1];
		const int batchSize = 1;
	    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
		for (int m = 0; m < batchSize; m++)
		{
			std::vector<itmCellRcgz_x100> out;
			float sum{ 0.0f };
			for (int i = 0; i < outputSize; i++)
			{
				output[m*outputSize + i] = exp(output[m*outputSize + i]);
				sum += output[m*outputSize + i];
			}
			bool bNedPush = false;
			for (int i = 0; i < outputSize; i++)
			{
				output[m*outputSize + i] /= sum;
				bNedPush = true;
				itmCellRcgz_x100 itm;
				itm.m_type = i;
				itm.m_pcnt = output[m*outputSize + i];
				for (size_t j = 0; j < out.size(); j++)
				{
					itmCellRcgz_x100& obj = out.at(j);
					if (itm.m_pcnt > obj.m_pcnt)
					{
						out.insert(out.begin() + j, itm);
						bNedPush = false;
						break;
					}
				}
				if (bNedPush && 0.0 < itm.m_pcnt)
				{
					out.push_back(itm);
				}
			}
			for (int i = 0; i < out.size(); i++)
				outList.push_back(out[i]);
		}	    		
	    return true;
	}

private:
    samplesCommon::SampleParams mParams; //!< The parameters for the sample.
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::IRuntime> mRuntime; 
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
	//!
	//! \brief Reads the input and stores the result in a managed buffer
	//!
    bool processInput(const samplesCommon::BufferManager& buffers, cv::Mat& src)
    {
	    const int inputC = 3;
	    const int inputH = 128;
	    const int inputW = 128;
	    //const int batchSize = 1;

	    cv::Scalar mean_(0.485, 0.456, 0.406);
	    cv::Scalar std_(0.229, 0.224, 0.225);
		float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
		cv::Mat image;
		cv::resize(src, image, cv::Size(128, 128));
		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

		image.convertTo(image, CV_32FC3);
		image = image / 255;
		cv::subtract(image, mean_, image);
		cv::divide(image, std_, image);

		// subtract image channel mean

		for (int row = 0; row < inputH; row++)
		{
			float* data = image.ptr<float>(row);
			for (int col = 0; col < inputW; col++)
			{
				for (int c = 0; c < inputC; c++)
				{
					hostDataBuffer[c * inputW*inputH + row * inputW + col] = data[col*inputC + c];
				}
			}
		}
		
	    
	    return true;
	}
};

