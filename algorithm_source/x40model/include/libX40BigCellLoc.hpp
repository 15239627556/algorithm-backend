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

//x40巨核细胞定位结果
struct  itmX40BigCellInfo
{
	int index;
	std::vector<cv::Rect> bigCellInfo;
	std::vector<float> bigCellRate;
};

typedef std::pair<float, float> ratio_;
typedef std::tuple<float, int, int> class_conf_idx_label;


class X40BigCellLocateOnnx
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	const int inputH = 512;
	const int inputW = 640;
	vector<float> image_shape{ 640, 512 };
	vector<vector<float> > anchor_boxes;
	// int device;
	X40BigCellLocateOnnx(int gpu_id) :sortThreadValue(0.5)
	{
		mParams.inputTensorNames.push_back("input");
		mParams.batchSize = 4;
		mParams.outputTensorNames.push_back("cls_scores");
		mParams.outputTensorNames.push_back("boxes");
		mParams.dlaCore = -1;
		mParams.int8 = false;
		mParams.fp16 = false;

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
			engine = "engines/2080/x40_bigcell_locate.trt";
		else if(index_3080 != string::npos)
			engine = "engines/3080/x40_bigcell_locate.trt";
		else if(index_4070 != string::npos)
			engine = "engines/4070/x40_bigcell_locate.trt";
		else if(index_4090 != string::npos)
			engine = "engines/4070/x40_bigcell_locate.trt";
		else
			std::cout << "cannot find correct trt" << std::endl;
		
		initLibNvInferPlugins(nullptr, "");
		// std::string engine = "engines/x40_bigcell_locate.trt";
		// std::string engine = "engines/4070/x40_bigcell_locate.trt";
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
		sample::gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kERROR);  // 设置日志
		mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
		mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
		
	}
	~X40BigCellLocateOnnx()
	{}
	bool infer(vector<cv::Mat> image, std::vector<itmX40BigCellInfo>& outlist)
	{
		if (!mEngine) {
			std::cout << "x40 big mEngine error" << std::endl;
			return false;
		}
		try{
			// std::cout << "x40 big context start" << std::endl;
			samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
			auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
			// std::cout << "x40 big context end" << std::endl;
			if (!context)
			{
				std::cout << "x40 big context error" << std::endl;
				return false;
			}
			// Read the input data into the managed buffers
			// assert(mParams.inputTensorNames.size() == 1);
			if (!processInput(buffers, image))
			{
				std::cout << "x40 big processInput error" << std::endl;
				return false;
			}

			//double time1 = static_cast<double>(cv::getTickCount());
			// Memcpy from host input buffers to device input buffers
			// cudaSetDevice(device);

			buffers.copyInputToDevice();
			// std::cout << "executeV2 start" << std::endl;
			bool status = context->executeV2(buffers.getDeviceBindings().data());
			if (!status)
			{
				std::cout << "x40 big executeV2 error" << std::endl;
				return false;
			}

			// Memcpy from device output buffers to host output buffers
			buffers.copyOutputToHost();

			
			float* rates = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
			float* rects = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));

			int batchSize = 4;
			// cout << _msize(regression) / sizeof(*regression) << endl;
			for (int b = 0; b < batchSize; b++)
			{
				vector<cv::Rect> localBoxes;
				vector<float> localConfidences;
				//cout << (_msize(rates) / sizeof(*rates)) << " " << (_msize(rects) / sizeof(*rects)) << endl;
				int len = 6720;
				for (int i = b * len; i < (b + 1) * len; i++)
				{
					// cout << i << " " << rates[i] << endl;
					if (rates[i] > 0.45)
					{
						/*cout << rects[i] << " " << rects[i * 4] << endl;*/
			
						localConfidences.push_back(rates[i]);
						localBoxes.push_back(cv::Rect(rects[i * 4] * image[b].cols / 612, rects[i * 4 + 1] * image[b].rows / inputH,
							rects[i * 4 + 2] * image[b].cols / 612 - rects[i * 4] * image[b].cols / 612,
							rects[i * 4 + 3] * image[b].rows / inputH - rects[i * 4 + 1] * image[b].rows / inputH));
					}
				}
				// NMS
				vector<int> nmsIndices;
				cv::dnn::NMSBoxes(localBoxes, localConfidences, 0.45f, 0.7f, nmsIndices);
				

				itmX40BigCellInfo tempInfo;
				tempInfo.index = b;
				tempInfo.bigCellRate = {};
				tempInfo.bigCellInfo = {};
				for (size_t idx = 0; idx < nmsIndices.size(); idx++) {
					size_t idx_ = nmsIndices[idx];
					/*out_boxes.push_back(localBoxes[idx_]);
					scores.push_back(localConfidences[idx_]);*/
					
					tempInfo.bigCellRate.push_back(localConfidences[idx_]);
					//外扩细胞框
					int x_new = localBoxes[idx_].x - localBoxes[idx_].width / 2;
					int y_new = localBoxes[idx_].y - localBoxes[idx_].height / 2;
					int w_new = localBoxes[idx_].width * 2;
					int h_new = localBoxes[idx_].height * 2;
					cv::Rect cell_rect_new = cv::Rect(x_new, y_new, w_new, h_new);

					tempInfo.bigCellInfo.push_back(cell_rect_new & cv::Rect(0, 0, image[b].cols, image[b].rows));
					
				}
				outlist.push_back(tempInfo);
			}
			return true;
		}catch (const std::exception& e) {
			sample::gLogError << "[infer] exception: " << e.what() << std::endl;
			return false;
    }
		
	}
private:
	samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.
    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; 
	float sortThreadValue; //用于筛选框选结果是否有效的判断值

	bool processInput(const samplesCommon::BufferManager& buffers, vector<cv::Mat> srclist)
	{
		// std::cout << "x40 big processInput start" << std::endl;
		const int inputC = 3;
		const int batchSize = 4;
		// cudaSetDevice(device); 
		float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	
		for (int m = 0; m < batchSize; m++)
		{
			cv::Mat src = srclist[m] + 0;
			// cv::resize(srclist[m], src, cv::Size(612, 512));
			cv::Mat image(inputH, inputW, CV_8UC3, cv::Scalar(114, 114, 114));
			image(cv::Rect(0, 0, src.cols, src.rows)) = src + 0;
			image.convertTo(image, CV_32FC3);

			for (int i = 0; i < inputH; i++) {
				float* data = image.ptr<float>(i);
				for (int j = 0; j < inputW; j++) {
					for (int k = 0; k < inputC; k++) {
						hostDataBuffer[m*inputH*inputW*inputC + k * inputW*inputH + i * inputW + j] = data[j*inputC + k];					
					}
				}
			}
		}
		// std::cout << "x40 big processInput end" << std::endl;
		return true;
	}
};



