#pragma once
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "opencv2/cudawarping.hpp"
#include <type_traits>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
using namespace cv;
using namespace std;

struct PicCellAnalysisResult{
	cv::Mat uOutPic;
	std::vector<cv::Point> cellCenterPoints;
};

class CellAnalysisOnnx
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	// int device;
	CellAnalysisOnnx(int gpu_id)
	{
		// cudaSetDevice(setdevice); 
		// device = setdevice;
		mParams.inputTensorNames.push_back("input");
		mParams.batchSize = 4;
		mParams.outputTensorNames.push_back("center");
		mParams.outputTensorNames.push_back("center_pool");
		mParams.outputTensorNames.push_back("size");
		mParams.outputTensorNames.push_back("sem_seg");
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
			engine = "engines/2080/x40_cellAnalysis.trt";
		else if(index_3080 != string::npos)
			engine = "engines/3080/x40_cellAnalysis.trt";
		else if(index_4070 != string::npos)
			engine = "engines/4070/x40_cellAnalysis.trt";
		else if(index_4090 != string::npos)
			engine = "engines/4070/x40_cellAnalysis.trt";
		else
			std::cout << "cannot find correct trt" << std::endl;
		// std::string engine = "engines/x40_cellAnalysis.trt";
		// std::string engine = "engines/4070/x40_cellAnalysis.trt";
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
	~CellAnalysisOnnx()
	{}

	bool infer(std::vector<cv::Mat> uImg, std::vector<PicCellAnalysisResult>& uOutImg)
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
		// cudaSetDevice(device); 
		buffers.copyInputToDevice();

		bool status = context->executeV2(buffers.getDeviceBindings().data());
		
		if (!status)
		{
			return false;
		}
		buffers.copyOutputToHost();

		const int batchSize = 4;
		float* pred_cls = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
		float* pool_cls = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));
		float* pred_size = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[2]));
		float *mask = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[3]));   
		
		//cout << buffers->size(mParams.outputTensorNames[0]) << endl;

		const int resH = 512;   // 512
		const int resW = 640;    // 640
		float* ptr_cls = pred_cls;
		float* ptr_pool_cls = pool_cls;

		uOutImg.clear();

		
		for (int b = 0; b < batchSize; b++)
		{
			PicCellAnalysisResult uOutImg_singel;
			cv::Mat bgMask = cv::Mat(resH, resW, CV_32FC1, mask + b * (3 * resH * resW));
			cv::Mat whiteMask = cv::Mat(resH, resW, CV_32FC1, mask + b * (3 * resH * resW) + resH * resW);
			cv::Mat redMask = cv::Mat(resH, resW, CV_32FC1, mask + b * (3 * resH * resW) + 2 * resH * resW);

			cv::Mat bgr;
			vector<cv::Mat> channels = { whiteMask , bgMask , redMask };
			cv::merge(channels, bgr);
			bgr = bgr * 255;
			bgr.convertTo(bgr, CV_8UC3);



			uOutImg_singel.uOutPic = bgr(cv::Rect(0, 0, 612, 512));

			vector<Rect> localBoxes;
			vector<float> localConfidences;
			for (int j = 0; j < 512 * 640; j++) {
				
				if (*ptr_cls == *ptr_pool_cls && (*ptr_cls) >= 0.2) {

					int center_x = j % 640;
					int center_y = j / 640;

					if (whiteMask.at<float>(center_y, center_x) < .3)
					{
						ptr_cls++;
						ptr_pool_cls++;
						continue;
					}

					
					// int size_offset_index_x = (center_x)+(640 * center_y) + (640 * 512 * 0) + (640 * 512 * 2 * b);
					// int size_offset_index_y = (center_x)+(640 * center_y) + (640 * 512 * 1) + (640 * 512 * 2 * b);

					/*int x_min = (center_x  - pred_size[size_offset_index_x] / 2);
																				
					int y_min = (center_y  - pred_size[size_offset_index_y] / 2);
																				
					int x_max = (center_x  + pred_size[size_offset_index_x] / 2);
																				
					int y_max = (center_y  + pred_size[size_offset_index_y] / 2);*/

					int x_min = (center_x - 25 / 2);
											
					int y_min = (center_y - 25 / 2);
											
					int x_max = (center_x + 25 / 2);
											
					int y_max = (center_y + 25 / 2);


					Rect box = Rect(x_min, y_min, x_max - x_min, y_max - y_min);
					localBoxes.push_back(box);
					localConfidences.push_back(*ptr_cls);

				}
				ptr_cls++;
				ptr_pool_cls++;

			}		
			vector<Rect> out_boxes;  
			vector<double> scores;  
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
		return true;
	}



private:
	samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.
	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
	int mNumber{ 0 };             //!< The number to classify
	std::shared_ptr<nvinfer1::IRuntime> mRuntime; 
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
	bool processInput(const samplesCommon::BufferManager& buffers, std::vector<cv::Mat> uImg)
	{
		const int inputC = 4;
		const int inputH = 512;
		const int inputW = 640;
		const int batchSize = 4;
		// cudaSetDevice(device); 
		float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
		for (int m = 0; m < batchSize; m++)
		{
			cv::Mat HSV;
			cv::cvtColor(uImg[m], HSV, cv::COLOR_BGR2HSV);
			std::vector<cv::Mat> channels;
			cv::split(HSV, channels);

			std::vector<cv::Mat> channels_;
			cv::split(uImg[m], channels_);
			channels_.push_back(channels.at(1));
			cv::merge(channels_, uImg[m]);
			uImg[m].convertTo(uImg[m], CV_32FC4);
			uImg[m] = uImg[m] / 255;

			for (int i = 0; i < inputH; i++) {
				float* data = uImg[m].ptr<float>(i);

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