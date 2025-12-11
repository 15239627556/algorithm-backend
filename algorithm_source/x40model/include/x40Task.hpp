#pragma once
#include <opencv2/opencv.hpp>
#include "libX40BigCellLoc.hpp"
#include "lib40XConstituencyScore.hpp"
#include "libCellAnalysis.hpp"
#include "libX40HaveCellLoc.hpp"
using namespace cv;
using namespace std;
#define PI  acos(-1)
#define INPUT_IMAGE_NUM     4
#define BLOCK_NUM           4

class X40Task
{
public:
    X40Task()
    {
        lpLocation40xBig = nullptr;
        lpLocation40xBig = new X40BigCellLocateOnnx();

        lpConstituency = nullptr;
	    lpConstituency = new X40ConstituencyOnnx();

        lpCellAnalysis = nullptr;
        lpCellAnalysis = new CellAnalysisOnnx();

        lpLocate40xHave = nullptr;
        lpLocate40xHave = new CellLocateOnnx();

        // 构建 anchor boxes（略去重复逻辑以节省篇幅）
        float anchor_scale = 2;
        vector<float> pyramid_levels{4, 5, 6};
        vector<float> strides;
        vector<ratio_> ratios{make_pair(1.0, 1.0), make_pair(1.2, 0.8), make_pair(0.8, 1.2)};
        vector<vector<ratio_>> anchor_size{
            {make_pair(44, 56), make_pair(58, 58), make_pair(56, 44)},
            {make_pair(56, 72), make_pair(76, 76), make_pair(72, 56)},
            {make_pair(90, 90), make_pair(112, 112), make_pair(128, 128)}};

        for (size_t i = 0; i < pyramid_levels.size(); i++)
        {
            strides.push_back(pow(2, pyramid_levels[i]));
        }

        for (size_t i = 0; i < strides.size(); i++)
        {
            int index_m = ceil((640 - strides[i] / 2) / strides[i]);
            int index_n = ceil((512 - strides[i] / 2) / strides[i]);
            for (int n = 0; n < index_n; n++)
            {
                for (int m = 0; m < index_m; m++)
                {
                    for (size_t k = 0; k < ratios.size(); k++)
                    {
                        float anchor_size_x_2 = anchor_size[i][k].first / 2;
                        float anchor_size_y_2 = anchor_size[i][k].second / 2;
                        int x = strides[i] / 2 + m * strides[i];
                        int y = strides[i] / 2 + n * strides[i];
                        vector<float> box{y - anchor_size_y_2, x - anchor_size_x_2,
                                          y + anchor_size_y_2, x + anchor_size_x_2};
                        static_cast<X40BigCellLocateOnnx *>(lpLocation40xBig)->anchor_boxes.push_back(box);
                    }
                }
            }
        }
    }

    ~X40Task()
    {
        delete lpLocation40xBig;
        lpLocation40xBig = nullptr;

        delete lpConstituency;
        lpConstituency = nullptr;

        delete lpCellAnalysis;
        lpCellAnalysis = nullptr;

        delete lpLocate40xHave;
        lpLocate40xHave = nullptr;
    }
    /*
        *************************************图片前处理*********************************
    */
    void imageProcessing1(vector<cv::Mat> uPicMatlist, vector<cv::Mat>& outPicMatlist)
    {
        outPicMatlist.clear();
        for(size_t i = 0; i < uPicMatlist.size(); i++)
        {
            cv::Mat src;
            cv::resize(uPicMatlist[i], src, cv::Size(612, 512));
            outPicMatlist.push_back(src);
        }
    }

    void imageProcessing2(vector<cv::Mat> uPicMatlist, vector<cv::Mat>& outPicMatlist)
    {
        const int inputH = 512;
        const int inputW = 640;

        outPicMatlist.clear();
        for(size_t i = 0; i < uPicMatlist.size(); i++)
        {
            cv::Mat uImg = uPicMatlist[i] + 0;
            cv::Mat gray;
            cv::cvtColor(uImg, gray, cv::COLOR_BGR2GRAY);
            cv::Scalar white_mean_gray = mean(gray);
            if (white_mean_gray[0] < 160)
            {
                cv::Mat flat = gray.reshape(1, 1) + 0;
                cv::sort(flat, flat, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
                int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * 0.85));
                cv::Mat white_mask = gray >= highval;
                cv::Scalar white_mean = mean(uImg, white_mask);

                std::vector<cv::Mat> channels;
                cv::split(uImg, channels);
                uImg.convertTo(uImg, CV_32FC3);
                channels.at(0) = channels.at(0) * (float(250) / white_mean[0]);
                channels.at(1) = channels.at(1) * (float(250) / white_mean[1]);
                channels.at(2) = channels.at(2) * (float(250) / white_mean[2]);
                cv::merge(channels, uImg);
                uImg.convertTo(uImg, CV_8UC3);
            }

            cv::Mat image(inputH, inputW, CV_8UC3, cv::Scalar(230, 230, 230));
            image(cv::Rect(0, 0, uImg.cols, uImg.rows)) = uImg + 0;
            outPicMatlist.push_back(image);
        }
    }

    /*
        *************************************图片前处理*********************************
    */

    /*
        *************************************有核细胞筛选********************************
    */
    void filter_cell_by_center_point(itemX40HaveLocateInfo X40HaveCellLocateOut, cv::Mat b, itemX40HaveLocateInfo& new_out)
    {
        for(size_t i = 0; i < X40HaveCellLocateOut.cellrects.size(); i++)
        {
            int x = X40HaveCellLocateOut.cellrects[i].x;
            int y = X40HaveCellLocateOut.cellrects[i].y;
            int w = X40HaveCellLocateOut.cellrects[i].width;
            int h = X40HaveCellLocateOut.cellrects[i].height;
            int p_x = (x + w / 2) / 4;
            int p_y = (y + h / 2) / 4;
            if(b.at<uchar>(p_y, p_x) > 0 && X40HaveCellLocateOut.types[i] == 0)
            {
                new_out.cellrects.push_back(X40HaveCellLocateOut.cellrects[i]);
                new_out.scores.push_back(X40HaveCellLocateOut.scores[i]);
                new_out.types.push_back(X40HaveCellLocateOut.types[i]);
            }
        }
        return ;
    }
    /*
        ******************************有核细胞筛选*********************************
    */

    /*
        ******************************计算区域评分**********************************
    */

    // 九宫格等级概率计算
	float nineClass_prod(float nineGrid_score)
	{
		float nineClass_v = 0.0f;
		if (nineGrid_score <= 64 && nineGrid_score > 45)
			nineClass_v = 8.5f * nineGrid_score * 0.001f;

		else if (nineGrid_score <= 45 && nineGrid_score > 24)
			nineClass_v = 5.5f * nineGrid_score * 0.001f;
		else
			nineClass_v = 0.16875f * nineGrid_score * 0.001f;

		return nineClass_v;
	}
	//有核细胞总面积归一化
	double total_MinMaxScaler(int x)
	{
		double data_min = 78.0;
		double data_max = 5604.0;
		double feature_range[2] = { 0.08, 5.6 };
		double scale = (feature_range[1] - feature_range[0]) / (data_max - data_min);
		double min_ = feature_range[0] - data_min * scale;
		double totalYouhe_y = scale * x + min_;
		return totalYouhe_y;
	}
	//有核细胞总面积概率计算: 服从卡方分布
	double totalArea_chi(double totalArea_scaler)
	{
		double k = 4.0;
		double scale = 0.55;
		totalArea_scaler = totalArea_scaler / scale;
		double a = 1 / ((pow(2, k / 2)) * 1);
		double b = pow(totalArea_scaler, (k / 2 - 1));
		double c = exp(-totalArea_scaler / 2);
		double totalArea_v = (a * b * c) / scale;
		return totalArea_v;
	}

	double totalArea_prod(int totalArea_Nucleated)
	{
		double totalArea_scaler = total_MinMaxScaler(totalArea_Nucleated);
		double totalArea_v = totalArea_chi(totalArea_scaler);
		if (totalArea_v < 0.0257)
			return  0.0257 + 0.1;

		else
			return totalArea_v + 0.1;
	}
	//红细胞面积归一化
	double red_MinMaxScaler(int x)
	{
		int data_min = 3076;
		int data_max = 17786;
		double feature_range[2] = { 3.0, 18.0 };
		double scale = (feature_range[1] - feature_range[0]) / (data_max - data_min);
		double min_ = feature_range[0] - data_min * scale;
		double red_y = scale * x + min_;
		return red_y;
	}
	// 红细胞总面积概率计算: 服从高斯分布
	double redArea_gauss(double redArea_scaler)
	{
		double loc = 9.120618882524237 - 0.5;
		double scale = 4.886065801133071 - 2.5;
		double a = (pow((redArea_scaler - loc), 2)) / (2 * pow(scale, 2));
		double Red_area_v = (1 / (scale * sqrt(2 * PI))) * exp(-a);
		return Red_area_v;
	}
	double redArea_prod(int Red_area)
	{

		double redArea_scaler = red_MinMaxScaler(Red_area);
		double Red_area_v = redArea_gauss(redArea_scaler);

		if (Red_area_v < 0.01)
			return 0.01 + 0.15;
		else
			return Red_area_v + 0.15;
	}
	/*计算九宫格分值*/
	float logPrior(float nineClass_v, int totalArea_v, int Red_area_v)
	{
		float logPrior_num = log(nineClass_prod(nineClass_v)) + log(float(totalArea_prod(totalArea_v))) + log(float(redArea_prod(Red_area_v)));
		return logPrior_num;
	}

    void confirm_block_grade_and_score(itmCellRcgzConstituencyBigImg blockScoreInfo, int &out_grade, float &out_score)
    {
        int block_top1 = blockScoreInfo.uBigData[0].m_type;
        int class_dict[7] = {64, 32, 16, 8, 4, 2, 1};
        if(block_top1 == 0)
        {
            float score_temp = 0.0f;
            for(size_t j = 0; j < blockScoreInfo.uBigData.size(); j++)
            {
                score_temp += blockScoreInfo.uBigData[j].m_pcnt * class_dict[blockScoreInfo.uBigData[j].m_type];
            }
            out_grade = 0;
            out_score = score_temp;
        }
        else if(block_top1 == 1)
        {
            float score_temp = 0.0f;
            for(size_t j = 0; j < blockScoreInfo.uBigData.size(); j++)
            {
                score_temp += blockScoreInfo.uBigData[j].m_pcnt * class_dict[blockScoreInfo.uBigData[j].m_type];
            }
            out_grade = 1;
            out_score = score_temp;
        }
        else if(block_top1 == 2)
        {
            float score_temp = 0.0f;
            for(size_t j = 0; j < blockScoreInfo.uBigData.size(); j++)
            {
                score_temp += blockScoreInfo.uBigData[j].m_pcnt * class_dict[blockScoreInfo.uBigData[j].m_type];
            }
            out_grade = 2;
            out_score = score_temp;
        }
        else if(block_top1 == 3)
        {
            out_grade = 3;
            out_score = class_dict[3];
        }
        else if(block_top1 == 4)
        {
            out_grade = 4;
            out_score = class_dict[4];
        }
        else if(block_top1 == 5)
        {
            out_grade = 5;
            out_score = class_dict[5];
        }
        else
        {
            out_grade = 6;
            out_score = class_dict[6];
        }
        return;
    }
    /*
        ******************************计算区域评分**********************************
    */

    /*
        ******************************调用模型接口**********************************
    */

    bool Location40X(vector<cv::Mat> uPicMatlist/*, TaskDataBlock *result*/)
    {
        if (!lpLocation40xBig || !lpConstituency || !lpCellAnalysis || !lpLocate40xHave) return false;
        for (auto &img : uPicMatlist)
            if (!img.data) return false;

        if(uPicMatlist.size() <= 0)
            return false;
        size_t actualNum = uPicMatlist.size();
        std::cout << "actualNum---->>> " << actualNum << std::endl;
        if(uPicMatlist.size() < INPUT_IMAGE_NUM)
        {
            //补全
            
            for(size_t i = actualNum; i < INPUT_IMAGE_NUM; i++)
                uPicMatlist.push_back(uPicMatlist[0]);
        }

        vector<cv::Mat> inputImages;
        imageProcessing1(uPicMatlist, inputImages);

        std::vector<itmX40BigCellInfo> x40BigCellLocateOut;
        lpLocation40xBig->infer(inputImages, x40BigCellLocateOut);
        std::vector<itmCellRcgzConstituencyBigImg> x40ConstituencyScoreOut;
        lpConstituency->infer(inputImages, x40ConstituencyScoreOut);
        std::vector<cv::Mat> x40CellAnalysisOut;
        vector<cv::Mat> inputImages_cellAnalysis;
        imageProcessing2(inputImages, inputImages_cellAnalysis);
        lpCellAnalysis->infer(inputImages_cellAnalysis, x40CellAnalysisOut);
        for(size_t b = 0; b < actualNum; b++)
        {
            itemX40HaveLocateInfo X40HaveCellLocateOut;
            lpLocate40xHave->infer(uPicMatlist[b], X40HaveCellLocateOut);

            std::vector<cv::Mat> channels;
            cv::split(x40CellAnalysisOut[b], channels);
            cv::Mat b_image = channels.at(0); //白细胞
            cv::Mat r_image = channels.at(2); //红细胞
            
            b_image = b_image > 200;
            r_image = r_image > 128;

            //过滤中心点不在白细胞区域内的细胞
            itemX40HaveLocateInfo X40HaveCellLocateOutFilter;
            filter_cell_by_center_point(X40HaveCellLocateOut, b_image, X40HaveCellLocateOutFilter);
            // std::cout << "X40HaveCellLocateOut.size()--->>> " << X40HaveCellLocateOut.cellrects.size() << "X40HaveCellLocateOutFilter.size()--->>>" << X40HaveCellLocateOutFilter.cellrects.size() << std::endl;
            //有核细胞信息赋值
            //result[b].result.imageResultInfos.haveCellCenterPointsSize = X40HaveCellLocateOutFilter.cellrects.size();
            for(size_t i = 0; i < X40HaveCellLocateOutFilter.cellrects.size(); i++)
            {
                int x = X40HaveCellLocateOutFilter.cellrects[i].x;
                int y = X40HaveCellLocateOutFilter.cellrects[i].y;
                int w = X40HaveCellLocateOutFilter.cellrects[i].width;
                int h = X40HaveCellLocateOutFilter.cellrects[i].height;
               // result[b].result.imageResultInfos.haveCellCenterPoints[i].c_x = x + w / 2;
               // result[b].result.imageResultInfos.haveCellCenterPoints[i].c_y = y + y / 2;
            }

            //计算区域分值
            int len = BLOCK_NUM;
            for(int bl = b *len; bl < (b + 1) * len; bl++)
            {
                cv::Rect block_rect = x40ConstituencyScoreOut[bl].uBigImg;
                int b_area = cv::countNonZero(b_image(block_rect));
                int r_area = cv::countNonZero(r_image(block_rect));
                //确定区域等级和分值
                int grade = 0;
                float score = 0.0f;
                confirm_block_grade_and_score(x40ConstituencyScoreOut[bl], grade, score);
                score = logPrior(score, b_area, r_area);

                // result[b].result.imageResultInfos.areaScoreInfo[bl % len].x = block_rect.x;
                // result[b].result.imageResultInfos.areaScoreInfo[bl % len].y = block_rect.y;
                // result[b].result.imageResultInfos.areaScoreInfo[bl % len].w = block_rect.width;
                // result[b].result.imageResultInfos.areaScoreInfo[bl % len].h = block_rect.height;
                // result[b].result.imageResultInfos.areaScoreInfo[bl % len].score = score;
            }
            //巨核细胞信息赋值
           // result[b].result.imageResultInfos.bigCellRectsSize = x40BigCellLocateOut[b].bigCellInfo.size();
            // for(int i = 0; i < x40BigCellLocateOut[b].bigCellInfo.size(); i++)
            // {
            //     result[b].result.imageResultInfos.bigCellRects[i].x = x40BigCellLocateOut[b].bigCellInfo[i].x;
            //     result[b].result.imageResultInfos.bigCellRects[i].y = x40BigCellLocateOut[b].bigCellInfo[i].y;
            //     result[b].result.imageResultInfos.bigCellRects[i].w = x40BigCellLocateOut[b].bigCellInfo[i].width;
            //     result[b].result.imageResultInfos.bigCellRects[i].h = x40BigCellLocateOut[b].bigCellInfo[i].height;
            // }
            
        }
        // std::cout << x40BigCellLocateOut.size() << " " << x40ConstituencyScoreOut.size() << endl;
        return true;
    }
/*
        ******************************调用模型接口**********************************
*/
private:
    X40BigCellLocateOnnx *lpLocation40xBig;
    X40ConstituencyOnnx *lpConstituency;
    CellAnalysisOnnx *lpCellAnalysis;
    CellLocateOnnx *lpLocate40xHave;
};