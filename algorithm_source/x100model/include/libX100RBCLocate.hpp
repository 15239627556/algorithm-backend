
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
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


//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!

class X100RBCLocateOnnx
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	X100RBCLocateOnnx(int gpu_id) :sortThreadValue(0.5)
	{
		cudaSetDevice(gpu_id);
		mParams.inputTensorNames.push_back("input");
		mParams.batchSize = 1;
		mParams.outputTensorNames.push_back("256");
		mParams.outputTensorNames.push_back("263");
		mParams.outputTensorNames.push_back("259");
		mParams.outputTensorNames.push_back("262");
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
			engine = "engines/2080/x100_RBC_locate.trt";
		else if(index_3080 != string::npos)
			engine = "engines/3080/x100_RBC_locate.trt";
		else if(index_4070 != string::npos)
			engine = "engines/4070/x100_RBC_locate.trt";
		else if(index_4090 != string::npos)
			engine = "engines/4070/x100_RBC_locate.trt";
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

		float anchor_scale = 2;  // constant !!!!!

		vector<float> pyramid_levels{ 4, 5, 6 }; // layers
		vector<float> strides;
		for (size_t i = 0; i < pyramid_levels.size(); i++) {
			int pow_ = pow(2, pyramid_levels[i]);
			strides.push_back(pow_);
		}
		vector<float> scales{ 1, 1.4142135623730951 }; // (2^0, 2^(1/3), 2^(2/3))
		vector<ratio_> ratios{ make_pair(1.0, 1.0), make_pair(1.2, 0.8), make_pair(0.8, 1.2) };
		for (size_t i = 0; i < strides.size(); i++) {
			int index_m = ceil((image_shape[1] - strides[i] / 2) / strides[i]);
			int index_n = ceil((image_shape[0] - strides[i] / 2) / strides[i]);
			vector<vector<int> > xv(index_n, vector<int>(index_m));
			vector<vector<int> > yv(index_n, vector<int>(index_m));
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
							vector<float> box{ yv[n][m] - anchor_size_y_2, xv[n][m] - anchor_size_x_2,
								yv[n][m] + anchor_size_y_2, xv[n][m] + anchor_size_x_2 };
							anchor_boxes.push_back(box);

						}
					}
				}
			}
		}
	}
	~X100RBCLocateOnnx()
	{
	}
	//!
	//! \brief Runs the TensorRT inference engine for this sample
	//!
	bool infer(cv::Mat& image, vector<cv::Rect>& out)
	{
		out.clear();
		samplesCommon::BufferManager buffers(mEngine);
    	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
		if (!context)
		{
			return false;
		}
		// std::cout << "1111" << std::endl;
		// Read the input data into the managed buffers
		assert(mParams.inputTensorNames.size() == 1);
		if (!processInput(buffers, image))
		{
			return false;
		}
		// std::cout << "2222" << std::endl;
		// Memcpy from host input buffers to device input buffers
		buffers.copyInputToDevice();

		bool status = context->executeV2(buffers.getDeviceBindings().data());
		if (!status)
		{
			return false;
		}
		// std::cout << "3333" << std::endl;
		// Memcpy from device output buffers to host output buffers
		buffers.copyOutputToHost();

		float* cls = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
		float* wh = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[2]));
		float* offset = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[3]));
		float* clsMax = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));

		int batchSize = 1;

		int outWidth = image.cols / 16;
		int outHeight = image.rows / 16;

		// std::cout << "image.cols == " << image.cols << " " << "image.rows == " << image.rows << std::endl;

		float threshold = 0.3;

		for (int i = 0; i < batchSize; i++) {
			// vector<itmRedCellLocInfo> instances{};
			int clsStartPos = i * outWidth * outHeight;
			cv::Mat clsMat = cv::Mat(outHeight, outWidth, CV_32FC1, cls + clsStartPos);
			cv::Mat clsMaxMat = cv::Mat(outHeight, outWidth, CV_32FC1, clsMax + i);
			cv::Mat mask = (clsMat > threshold) & (abs(clsMat - clsMaxMat) < 1e-6);
			vector<cv::Point> locations;
			cv::findNonZero(mask, locations);
			// cv::imwrite("mask.jpg", mask);
			// cv::imwrite("clsMat.jpg", clsMat);
			// cv::imwrite("clsMaxMat.jpg", clsMaxMat);

			cv::Mat xOffsetMat = cv::Mat(outHeight, outWidth, CV_32FC1,
				offset + clsStartPos * 2);
			cv::Mat yOffsetMat = cv::Mat(outHeight, outWidth, CV_32FC1,
				offset + clsStartPos * 2 + outWidth * outHeight);
			cv::Mat wMat = cv::Mat(outHeight, outWidth, CV_32FC1, wh + clsStartPos * 2);
			cv::Mat hMat = cv::Mat(outHeight, outWidth, CV_32FC1,
				wh + clsStartPos * 2 + outWidth * outHeight);

			// std::cout << "locations.size() == " << locations.size() << std::endl;
			for (cv::Point loc : locations) {
				// itmRedCellLocInfo instance;
				int x = loc.x;
				int y = loc.y;
				float centerX = ((float)x + xOffsetMat.at<float>(loc)) / 128 * 2048;
				float centerY = ((float)y + yOffsetMat.at<float>(loc)) / 96 * 1536;
				float width = wMat.at<float>(loc) / 128 * 2048;
				float height = hMat.at<float>(loc) / 96 * 1536;
				// instance.cellRectInfo = Rect((int)(centerX - width / 2), (int)(centerY - height / 2), (int)width, (int)height);
				// instance.cellRate = clsMat.at<float>(loc);
				cv::Rect cellRect = Rect((int)(centerX - width / 2), (int)(centerY - height / 2), (int)width, (int)height);
				out.push_back(cellRect & cv::Rect(0, 0, image.cols, image.rows));
			}
		}

		return true;
	}

private:
	samplesCommon::SampleParams mParams; //!< The parameters for the sample.
	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
	int mNumber{ 0 };             //!< The number to classify
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
	std::shared_ptr<nvinfer1::IRuntime> mRuntime; 
	float sortThreadValue; 
	vector<vector<float> > anchor_boxes;
	vector<float> image_shape{ 384,512 };
	//!
	//! \brief Parses an ONNX model for MNIST and creates a TensorRT network
	//!
	//!
	bool processInput(const samplesCommon::BufferManager& buffers, cv::Mat& src)
	{
		const int inputC = 3;
		const int inputH = 384;
		const int inputW = 512;
		const int batchSize = 1;

		cv::Scalar mean_(123.675, 116.28, 103.53);
		cv::Scalar std_(58.395, 57.12, 57.375);
		cv::Mat image;
		cv::resize(src, image, cv::Size(inputW, inputH));
		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
		image.convertTo(image, CV_32FC3);

		cv::subtract(image, mean_, image);
		cv::divide(image, std_, image);

		// subtract image channel mean
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

