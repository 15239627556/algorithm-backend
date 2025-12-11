#pragma once
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "opencv2/cudawarping.hpp"
#include <type_traits>
using namespace cv;
using namespace std;

struct itemX40HaveLocateInfo
{
    std::vector<cv::Rect> cellrects;
    std::vector<float> scores;
    std::vector<int> types;
};

struct  itmCellInfo
{
	cv::Rect rect;
	int celltype;
	float score;
	int area;
	bool isDel;
};

bool compareByArea(const itmCellInfo& a, const itmCellInfo& b) {
	return a.rect.width * a.rect.height > b.rect.width * b.rect.height;  
}


class CellLocateOnnx
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	CellLocateOnnx(int gpu_id)
	{
		mParams.inputTensorNames.push_back("images");
		mParams.batchSize = 9;
		mParams.outputTensorNames.push_back("output0");
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
			engine = "engines/2080/x40_have_cell_locate.trt";
		else if(index_3080 != string::npos)
			engine = "engines/3080/x40_have_cell_locate.trt";
		else if(index_4070 != string::npos)
			engine = "engines/4070/x40_have_cell_locate.trt";
		else if(index_4090 != string::npos)
			engine = "engines/4070/x40_have_cell_locate.trt";
		else
			std::cout << "cannot find correct trt" << std::endl;

		// std::string engine = "engines/x40_have_cell_locate.trt";
		// std::string engine = "engines/4070/x40_have_cell_locate.trt";
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
	

	~CellLocateOnnx()
	{}
	
	bool infer(cv::Mat uImg, itemX40HaveLocateInfo& uOutX40HaveCellLocate)
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
		
		std::vector<cv::Rect> block_rects;
		if (!processInput(buffers, uImg, block_rects))
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

		const int batchSize = 9;
		const int picNum = 1;
		float* data = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

        // int outputIndex = mEngine->getBindingIndex(mParams.outputTensorNames[0].c_str());
        // nvinfer1::Dims outputDims = mEngine->getBindingDimensions(outputIndex);
        // int totalOutputSize = 1;
        // for (int i = 0; i < outputDims.nbDims; ++i) totalOutputSize *= outputDims.d[i];
        // std::cout << "Output size = " << totalOutputSize << std::endl;


		for (int pic = 0; pic < picNum; pic++)
		{
			std::vector<itmCellInfo> cellitems;
			for (int b = 0; b < batchSize; b++)
			{
				std::vector<cv::Rect> localBoxes;
				std::vector<float> localConfidences;
				std::vector<int> celltypes;
				
				int image_start_index = pic * batchSize * 17661 * 6 + b * 17661 * 6;


				for (int cell = 0; cell < 17661; cell++)
				{
					int cell_x = data[image_start_index + 17661 * 0 + cell];
					int cell_y = data[image_start_index + 17661 * 1 + cell];
					int cell_w = data[image_start_index + 17661 * 2 + cell];
					int cell_h = data[image_start_index + 17661 * 3 + cell];
					float p1 = data[image_start_index + 17661 * 4 + cell];
					float p2 = data[image_start_index + 17661 * 5 + cell];


					int cls = 0;
					if (p2 > p1)
					{
						p1 = p2;
						cls = 1;
					}
					if (p1 > 0.5)
					{
                        // std::cout << cell_x << " "<< cell_y << " "<< cell_w << " "<< cell_h << " " << endl;
                        // std::cout << p1 << endl;
                        // std::cout << cls << endl;
						localBoxes.push_back(cv::Rect(cell_x , cell_y, cell_w , cell_h));
						localConfidences.push_back(p1);
						celltypes.push_back(cls);
					}
				}
				// NMS 
				std::vector<int> nmsIndices;
				cv::dnn::NMSBoxes(localBoxes, localConfidences, 0.5f, 0.45f, nmsIndices);

				vector<Rect> block_out_boxes;  
				vector<double> block_scores; 
				vector<int> cell_types;   
				for (size_t idx = 0; idx < nmsIndices.size(); idx++) {

					block_out_boxes.push_back(localBoxes[nmsIndices[idx]]);
					block_scores.push_back(localConfidences[nmsIndices[idx]]);
					cell_types.push_back(celltypes[nmsIndices[idx]]);
				}
				
				cv::Rect block_rect = block_rects[b];
				for (int cell = 0; cell < block_out_boxes.size(); cell++)
				{
					block_out_boxes[cell].x += block_rect.x;
					block_out_boxes[cell].y += block_rect.y;

					block_out_boxes[cell].x -= block_out_boxes[cell].width / 2;
					block_out_boxes[cell].y -= block_out_boxes[cell].height / 2;

					itmCellInfo cellitem;
					cellitem.celltype = cell_types[cell];
					cellitem.rect = block_out_boxes[cell];
					cellitem.score = block_scores[cell];
					cellitem.area = cellitem.rect.width * cellitem.rect.height;
					cellitem.isDel = false;
					cellitems.push_back(cellitem);
				}
			}
			
			std::sort(cellitems.begin(), cellitems.end(), compareByArea);

			for (int cell1 = 0; cell1 < cellitems.size(); cell1++)
			{
				if (cellitems[cell1].isDel)
					continue;

				uOutX40HaveCellLocate.cellrects.push_back(cellitems[cell1].rect);
				uOutX40HaveCellLocate.scores.push_back(cellitems[cell1].score);
				uOutX40HaveCellLocate.types.push_back(cellitems[cell1].celltype);
				//uOutImg.cellCenterPoints.push_back(cv::Point(cellitems[cell1].rect.x + cellitems[cell1].rect.width / 2, cellitems[cell1].rect.y + cellitems[cell1].rect.height / 2));


				if (cell1 == cellitems.size() - 1)
					continue;

				for (int cell2 = cell1 + 1; cell2 < cellitems.size(); cell2++)
				{
					if (cellitems[cell2].isDel)
						continue;

					int and_area = (cellitems[cell1].rect & cellitems[cell2].rect).area();
					float ratio = float(and_area) / cellitems[cell2].rect.area();
					if (ratio > 0.45) 
					{
						cellitems[cell2].isDel = true;
					}
				}
			}
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
	bool processInput(const samplesCommon::BufferManager& buffers, cv::Mat uImg, std::vector<cv::Rect>& block_rects)
	{
		
		const int inputC = 3;
		const int inputH = 928;
		const int inputW = 928;
		const int batchSize = 1;
		// subtract image channel mean
		//double time1 = static_cast<double>(cv::getTickCount());
        float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
        std::vector<cv::Mat> images;
		for (int m = 0; m < batchSize; m++)
		{
			cv::Mat image;
			uImg.convertTo(image, CV_32FC3);
			image = image / 255;

			int u_row_num = 3;
			int u_col_num = 3;
			double overlap_w = inputW * 0.1; //
			double overlap_h = inputW * 0.1; //
			for (int row = 0; row < u_row_num; row ++)
			{
				for (int col = 0; col < u_col_num; col ++)
				{
					int x_start = (inputW - overlap_w) * col;
					if(col == u_col_num - 1)
						x_start = uImg.cols - inputW;
					int y_start = (inputH - overlap_h) * row;
					if(row == u_row_num - 1)
						y_start = uImg.rows - inputH;
					images.push_back(image(cv::Rect(x_start, y_start, inputW, inputH)));
					block_rects.push_back(cv::Rect(x_start, y_start, inputW, inputH));
				}
			}

			
		}
        for (int h = 0; h < images.size(); h++)
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
		//time1 = ((double)cv::getTickCount() - time1) / cv::getTickFrequency();
		//cout << "processInput	" << time1 << std::endl;
		return true;
	}
};