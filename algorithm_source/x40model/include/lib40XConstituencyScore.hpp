
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
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
using namespace std;
#define celAbs(x,y) (x>y?x-y:y-x)

typedef struct
{
	int m_type;
	double m_pcnt;
}itmCellRcgzConstituency;

struct  itmCellRcgzConstituencyBigImg
{
	cv::Rect uBigImg;
	std::vector<itmCellRcgzConstituency> uBigData;
};

class X40ConstituencyOnnx
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
public:
	// int device;
	X40ConstituencyOnnx(int gpu_id) : mEngine(nullptr)
	{
		// cudaSetDevice(setdevice); 
		// device = setdevice;
		mParams.inputTensorNames.push_back("input.1");
		mParams.batchSize = 16;
		mParams.outputTensorNames.push_back("533");
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
			engine = "engines/2080/x40_constituency.trt";
		else if(index_3080 != string::npos)
			engine = "engines/3080/x40_constituency.trt";
		else if(index_4070 != string::npos)
			engine = "engines/4070/x40_constituency.trt";
		else if(index_4090 != string::npos)
			engine = "engines/4070/x40_constituency.trt";
		else
			std::cout << "cannot find correct trt" << std::endl;
		
		// std::string engine = "engines/4070/x40_constituency.trt";
		// std::string engine = "engines/x40_constituency.trt";
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
	~X40ConstituencyOnnx()
	{}

	
	bool infer(std::vector<cv::Mat> Bigimage, std::vector<itmCellRcgzConstituencyBigImg>& out)
	{
		samplesCommon::BufferManager buffers(mEngine);
    	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
		if (!context)
		{
			return false;
		}
		out.clear();
		std::vector<itmCellRcgzConstituency> uNeedData;
		assert(mParams.inputTensorNames.size() == 1);
		if (!processInput(buffers, Bigimage))
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
		const int outputSize = 7;
		const int batchSize = 16;
		float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
		// float val{ 0.0f };
		// int idx{ 0 };
		for (int b = 0; b < batchSize; b++)
		{
			float sum{ 0.0f };
			for (int i = b * outputSize; i < (b + 1)*outputSize; i++)
			{
				output[i] = exp(output[i]);
				// float f = output[i];
				sum += output[i];
			}

			bool bNedPush = false;
			uNeedData.clear();
			int uIndex = 0;
			for (int i = b * outputSize; i < (b + 1)*outputSize; i++)
			{
				output[i] /= sum;
				bNedPush = true;
				itmCellRcgzConstituency itm;
				itm.m_type = uIndex;
				uIndex++;
				itm.m_pcnt = output[i];
				for (size_t i = 0; i < uNeedData.size(); i++)
				{
					itmCellRcgzConstituency& obj = uNeedData.at(i);
					if (itm.m_pcnt > obj.m_pcnt)
					{
						uNeedData.insert(uNeedData.begin() + i, itm);
						bNedPush = false;
						break;
					}
				}
				if (bNedPush && 0 < itm.m_pcnt)
				{
					uNeedData.push_back(itm);
				}
			}
			if (uNeedData.size() > 0 && m_uBigInfo.size() > 0 && uNeedData.size() == uNeedData.size())
			{
				m_uBigInfo[b].uBigData = uNeedData;
			}
		}
		out = m_uBigInfo;
		/*for (int k = 0; k < 4; k++)
		{
			for(int kk = 0; kk < out[k].uBigData.size(); kk++)
				cout << out[k].uBigImg << " " << out[k].uBigData[kk].m_type << " " << out[k].uBigData[kk].m_pcnt << endl;
		}*/
		m_uBigInfo.clear();
		return true;
	}

private:
	std::vector<itmCellRcgzConstituencyBigImg> m_uBigInfo;
	samplesCommon::OnnxSampleParams mParams; 
	std::shared_ptr<nvinfer1::IRuntime> mRuntime; 
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
	nvinfer1::Dims mInputDims; 
	nvinfer1::Dims mOutputDims; 
	int mNumber{ 0 };
	
	bool processInput(const samplesCommon::BufferManager& buffers, std::vector<cv::Mat> src)
	{
		// cudaSetDevice(device); 
		m_uBigInfo.clear();
		const int inputC = 3;
		const int inputH = 224;
		const int inputW = 224;
		const int batchSize = 16;
		const int picNum = 4;
		cv::Scalar mean_(0.485, 0.456, 0.406);
		cv::Scalar std_(0.229, 0.224, 0.225);

		std::vector<cv::Mat> images;

		float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
		for (int m = 0; m < picNum; m++)
		{
			cv::Mat image;
			//cv::cvtColor(src[m], image, cv::COLOR_BGR2RGB);
			src[m].convertTo(image, CV_32FC3);
			image = image / 255;
			cv::subtract(image, mean_, image);
			cv::divide(image, std_, image);
			std::vector<cv::Rect> uImgRect;
			int image_size = 224;
			int uWIndex = src[m].cols / 612;
			int uHIndex = src[m].rows / 512;


			int w_skip = 212;
			int h_skip = 200;
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					images.push_back(image(cv::Rect(j*w_skip, (i*h_skip + 88), image_size, image_size)));
					uImgRect.push_back(cv::Rect(j*w_skip * uWIndex, (i*h_skip + 88) * uHIndex, image_size*uWIndex, image_size*uHIndex));
				}
			}
		

			for (size_t k = 0; k < uImgRect.size(); k++)
			{
				itmCellRcgzConstituencyBigImg uitmConstituencyBigImg;
				uitmConstituencyBigImg.uBigImg = uImgRect[k];
				m_uBigInfo.push_back(uitmConstituencyBigImg);
			}
			
		}
		for (int h = 0; h < batchSize; h++)
		{
			for (int i = 0; i < inputH; i++)
			{
				float* data = images[h].ptr<float>(i);
				for (int j = 0; j < inputW; j++)
				{
					for (int k = 0; k < inputC; k++)
					{
						hostDataBuffer[h*inputH*inputW*inputC + k * inputW*inputH + i * inputW + j] = data[j*inputC + k];
					}
				}
			}
		}
		
		return true;
	}
};


// bool dll_X40ConstituencyScore::Constituency40x(std::vector<cv::Mat> uBigPicMat, itmConstituencyBigImg *uOutValue)
// {
// 	libOnnxMNISTConstituency*lpclass = (libOnnxMNISTConstituency*)lpConstituency;
// 	if (!lpclass)
// 		return false;
// 	for (int i = 0; i < uBigPicMat.size(); i++)
// 	{
// 		if (NULL == uBigPicMat[i].data)
// 			return false;
// 	}
	
// 	int block_num = 4;
// 	std::vector<itmCellRcgzConstituencyBigImg> uInfo;
// 	int class_dict[7] = { 64, 32, 16, 8, 4, 2, 1 };
// 	if (!lpclass->inferBigImg(uBigPicMat, uInfo))
// 	{
// 		return false;
// 	}
// 	for (int i = 0; i < uInfo.size(); i++)
// 	{
// 		if (uInfo[i].uBigData.size() > 0)
// 		{
// 			itmConstituencyBigImg uPicRect;

// 			//0�� 
// 			if (uInfo[i].uBigData[0].m_type == 0)
// 			{
// 				uPicRect.uInfo.uBigImg.x = uInfo[i].uBigImg.x * 4;
// 				uPicRect.uInfo.uBigImg.y = uInfo[i].uBigImg.y * 4;
// 				uPicRect.uInfo.uBigImg.width = uInfo[i].uBigImg.width * 4;
// 				uPicRect.uInfo.uBigImg.height = uInfo[i].uBigImg.height * 4;

// 				float score_temp = 0.0;
// 				for (int j = 0; j < uInfo[i].uBigData.size(); j++)
// 				{
// 					score_temp += uInfo[i].uBigData[j].m_pcnt * class_dict[uInfo[i].uBigData[j].m_type];
// 				}

// 				uPicRect.uInfo.uScore = score_temp;
// 				uPicRect.uInfo.grade = 0;
// 				uPicRect.uPicIndex = i / block_num;
// 				uOutValue[i] = uPicRect;
// 				//uOutValue.push_back(uPicRect);
// 			}
// 			//1�� 
// 			else if (uInfo[i].uBigData[0].m_type == 1)
// 			{
// 				uPicRect.uInfo.uBigImg.x = uInfo[i].uBigImg.x * 4;
// 				uPicRect.uInfo.uBigImg.y = uInfo[i].uBigImg.y * 4;
// 				uPicRect.uInfo.uBigImg.width = uInfo[i].uBigImg.width * 4;
// 				uPicRect.uInfo.uBigImg.height = uInfo[i].uBigImg.height * 4;
// 				float score_temp = 0.0;
// 				for (int j = 0; j < uInfo[i].uBigData.size(); j++)
// 				{
// 					score_temp += uInfo[i].uBigData[j].m_pcnt * class_dict[uInfo[i].uBigData[j].m_type];
// 				}

// 				uPicRect.uInfo.uScore = score_temp;
// 				uPicRect.uInfo.grade = 1;
// 				uPicRect.uPicIndex = i / block_num;
// 				uOutValue[i] = uPicRect;
				
// 				//uOutValue.push_back(uPicRect);
// 			}
// 			//2�� 
// 			else if (uInfo[i].uBigData[0].m_type == 2)
// 			{
// 				uPicRect.uInfo.uBigImg.x = uInfo[i].uBigImg.x * 4;
// 				uPicRect.uInfo.uBigImg.y = uInfo[i].uBigImg.y * 4;
// 				uPicRect.uInfo.uBigImg.width = uInfo[i].uBigImg.width * 4;
// 				uPicRect.uInfo.uBigImg.height = uInfo[i].uBigImg.height * 4;
// 				float score_temp = 0.0;
// 				for (int j = 0; j < uInfo[i].uBigData.size(); j++)
// 				{
// 					score_temp += uInfo[i].uBigData[j].m_pcnt * class_dict[uInfo[i].uBigData[j].m_type];
// 				}

// 				uPicRect.uInfo.uScore = score_temp;
// 				uPicRect.uInfo.grade = 2;
// 				uPicRect.uPicIndex = i / block_num;
// 				uOutValue[i] = uPicRect;
				
// 				//uOutValue.push_back(uPicRect);
// 			}
// 			//3�� 
// 			else if (uInfo[i].uBigData[0].m_type == 3)
// 			{
// 				uPicRect.uInfo.uBigImg.x = uInfo[i].uBigImg.x * 4;
// 				uPicRect.uInfo.uBigImg.y = uInfo[i].uBigImg.y * 4;
// 				uPicRect.uInfo.uBigImg.width = uInfo[i].uBigImg.width * 4;
// 				uPicRect.uInfo.uBigImg.height = uInfo[i].uBigImg.height * 4;
// 				uPicRect.uInfo.uScore = class_dict[3];
// 				uPicRect.uInfo.grade = 3;
// 				uPicRect.uPicIndex = i / block_num;
// 				uOutValue[i] = uPicRect;
				
// 				//uOutValue.push_back(uPicRect);
// 			}
// 			//4��
// 			else if (uInfo[i].uBigData[0].m_type == 4)
// 			{
// 				uPicRect.uInfo.uBigImg.x = uInfo[i].uBigImg.x * 4;
// 				uPicRect.uInfo.uBigImg.y = uInfo[i].uBigImg.y * 4;
// 				uPicRect.uInfo.uBigImg.width = uInfo[i].uBigImg.width * 4;
// 				uPicRect.uInfo.uBigImg.height = uInfo[i].uBigImg.height * 4;
// 				uPicRect.uInfo.uScore = class_dict[4];
// 				uPicRect.uInfo.grade = 4;
// 				uPicRect.uPicIndex = i / block_num;
// 				uOutValue[i] = uPicRect;
				
// 				//uOutValue.push_back(uPicRect);
// 			}
// 			//5�� 
// 			else if (uInfo[i].uBigData[0].m_type == 5)
// 			{
// 				uPicRect.uInfo.uBigImg.x = uInfo[i].uBigImg.x * 4;
// 				uPicRect.uInfo.uBigImg.y = uInfo[i].uBigImg.y * 4;
// 				uPicRect.uInfo.uBigImg.width = uInfo[i].uBigImg.width * 4;
// 				uPicRect.uInfo.uBigImg.height = uInfo[i].uBigImg.height * 4;
// 				uPicRect.uInfo.uScore = class_dict[5];
// 				uPicRect.uInfo.grade = 5;
// 				uPicRect.uPicIndex = i / block_num;
// 				uOutValue[i] = uPicRect;

// 				//uOutValue.push_back(uPicRect);
// 			}
// 			//6�� 
// 			else if (uInfo[i].uBigData[0].m_type == 6)
// 			{
// 				uPicRect.uInfo.uBigImg.x = uInfo[i].uBigImg.x * 4;
// 				uPicRect.uInfo.uBigImg.y = uInfo[i].uBigImg.y * 4;
// 				uPicRect.uInfo.uBigImg.width = uInfo[i].uBigImg.width * 4;
// 				uPicRect.uInfo.uBigImg.height = uInfo[i].uBigImg.height * 4;
// 				uPicRect.uInfo.uScore = class_dict[6];
// 				uPicRect.uInfo.grade = 6;
// 				uPicRect.uPicIndex = i / block_num;
// 				uOutValue[i] = uPicRect;

// 				//uOutValue.push_back(uPicRect);
// 			}
			
// 		}
// 	}
// 	return true;
// }
