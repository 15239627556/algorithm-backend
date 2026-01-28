#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include "publicTRT.hpp"

struct LetterboxResult {
	cv::Mat img;
	float r;
	cv::Point pad;
};
template <typename T>
T clamp(T v, T lo, T hi) {
	return (v < lo) ? lo : (v > hi ? hi : v);
}

class CSFCellLocateOnnx: public PublicTRT
{

public:
	static inline const int INPUTC = 3;
	static inline const int INPUTH = 1024;
	static inline const int INPUTW = 1248;
	static inline const int BATCH = 1;

	CSFCellLocateOnnx(int gpu_id)
	{
		initTRT();
        std::string enginePath = selectEnginePath(gpu_id, "csf_locate");
        if (!loadEngine(enginePath)) {
            sample::gLogError << "Failed to load engine: " << enginePath << std::endl;
        }	
	}
	bool infer(std::vector<cv::Mat> uImgs, std::vector<itemX100CSFLocateInfo>& uOutValue)
	{
		rs.clear();
		pads.clear();
		std::vector<float> inputData = processInput(uImgs);
        
        // 调用基类执行推理
        if (!doInference({{"images", inputData}})) return false;

        // 获取输出
        float* ptr_data = mHostOutputs["output0"].data();

        postprocess(ptr_data, uOutValue);
        return true;
	}


private:
	int B = 1;
	int A = 5;
	int N = 106080;

	const float conf = 0.30f;
	const float iou = 0.50f;
	const int   maxd = 2000;
	const bool  agn = true;
	std::vector<float> rs;
	std::vector<cv::Point> pads;

	LetterboxResult letterbox(const cv::Mat& im,
		cv::Size new_shape = cv::Size(1248, 1248), float r = 0.5,
		const cv::Scalar& color = cv::Scalar(114, 114, 114),
		bool auto_shape = true,
		bool scale_fill = false,
		bool scaleup = true,
		bool center = true,
		int stride = 32,
		int interpolation = cv::INTER_LINEAR) 
	{
		assert(!im.empty());
		const int h = im.rows, w = im.cols;
		const int new_w = new_shape.width;
		const int new_h = new_shape.height;

		cv::Size new_unpad(std::lround(w * r), std::lround(h * r));

		float dw = static_cast<float>(new_w - new_unpad.width);
		float dh = static_cast<float>(new_h - new_unpad.height);

		if (scale_fill) 
		{

			new_unpad = cv::Size(new_w, new_h);
			dw = dh = 0.0f;
		}
		else if (auto_shape) 
		{
			dw = std::fmod(dw, static_cast<float>(stride));
			dh = std::fmod(dh, static_cast<float>(stride));
		}

		if (center) { dw *= 0.5f; dh *= 0.5f; }

		cv::Mat resized;
		if (new_unpad.width != w || new_unpad.height != h) 
		{
			cv::resize(im, resized, new_unpad, 0, 0, interpolation);
		}
		else 
		{
			resized = im.clone();
		}

		const int top = static_cast<int>(std::round(dh - 0.1f));
		const int bottom = static_cast<int>(std::round(dh + 0.1f));
		const int left = static_cast<int>(std::round(dw - 0.1f));
		const int right = static_cast<int>(std::round(dw + 0.1f));

		cv::Mat out;
		cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, color);

		return LetterboxResult{ std::move(out), r, cv::Point(left, top) };
	}
	cv::Mat makePredMatView(float* pred, int B, int A, int N)
	{
		int sizes[3] = { B, A, N };

		size_t steps[3] =
		{
			(size_t)A * N * sizeof(float),
			(size_t)N * sizeof(float),
			sizeof(float)
		};

		return cv::Mat(3, sizes, CV_32F, (void*)pred, steps);
	}

	void iou_and_overlap_xyxy(const float* a, const float* b,
		float& iou, float& overlap)
	{
		// a, b: [x1, y1, x2, y2]
		float ax1 = a[0], ay1 = a[1], ax2 = a[2], ay2 = a[3];
		float bx1 = b[0], by1 = b[1], bx2 = b[2], by2 = b[3];

		// ����
		float ix1 = std::max(ax1, bx1);
		float iy1 = std::max(ay1, by1);
		float ix2 = std::min(ax2, bx2);
		float iy2 = std::min(ay2, by2);
		float iw = std::max(0.0f, ix2 - ix1);
		float ih = std::max(0.0f, iy2 - iy1);
		float inter = iw * ih;

		// ���
		float areaA = std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);
		float areaB = std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);

		const float eps = 1e-6f;
		float uni = areaA + areaB - inter + eps;
		iou = inter / uni;

		float denom = std::max(std::min(areaA, areaB), eps);
		overlap = inter / denom;
	}

	std::vector<int> nms_numpy_like_with_dedup(
		const cv::Mat& boxes_xyxy,               // N x 4, CV_32F
		const std::vector<float>& scores,        // N
		float iou_thr,
		int   max_det,
		bool  extra_dedup = false,
		float overlap_thr = 0.7f,
		const std::vector<int>* orig_idx = nullptr // ���ǿգ��Ѿֲ��±�ӳ��ԭʼ�±�
	)
	{
		const int n = boxes_xyxy.rows;
		if (n == 0) return {};

		// ���ŶȽ���
		std::vector<int> order(n);
		std::iota(order.begin(), order.end(), 0);
		std::sort(order.begin(), order.end(),
			[&](int i, int j) { return scores[i] > scores[j]; });

		std::vector<int> kept_local;
		kept_local.reserve(std::min(max_det, n));
		std::vector<char> suppressed(n, 0);

		for (int oi = 0; oi < n && (int)kept_local.size() < max_det; ++oi)
		{
			int i = order[oi];
			if (suppressed[i]) continue;

			kept_local.push_back(i);
			const float* bi = boxes_xyxy.ptr<float>(i);

			for (int oj = oi + 1; oj < n; ++oj)
			{
				int j = order[oj];
				if (suppressed[j]) continue;

				const float* bj = boxes_xyxy.ptr<float>(j);
				float iou, overlap;
				iou_and_overlap_xyxy(bi, bj, iou, overlap);

				bool bad = (iou > iou_thr);
				if (extra_dedup) bad = bad || (overlap > overlap_thr);
				if (bad) suppressed[j] = 1;
			}
		}

		// ӳ��ԭʼ�±�
		if (orig_idx && (int)orig_idx->size() == n)
		{
			std::vector<int> kept;
			kept.reserve(kept_local.size());
			for (int li : kept_local) kept.push_back((*orig_idx)[li]);
			return kept;
		}
		else
		{
			return kept_local;  // δ�ṩӳ��ͷ��ؾֲ��±�
		}
	}

	std::vector<int> yolov_like_nms_cpp(
		const cv::Mat& xyxy_in,           // N x 4, CV_32F
		const std::vector<float>& conf_in,// N
		const std::vector<int>&   cls_in, // N
		float iou_thr,
		int   max_det,
		bool  agnostic,
		float max_wh = 7680.0f,
		int   max_nms = 30000,
		bool  extra_dedup = false,
		float overlap_thr = 0.70f,
		const std::vector<int>* classes = nullptr   // ��ѡ��������
	)
	{
		int n = xyxy_in.rows;
		if (n == 0) return {};

		// ---- 1) ��ѡ���������ˣ�ͬʱ����ԭʼ����ӳ��
		std::vector<int> orig_idx;
		orig_idx.reserve(n);

		cv::Mat xyxy = xyxy_in;                  // ������
		std::vector<float> conf = conf_in;
		std::vector<int>   cls = cls_in;

		if (classes && !classes->empty())
		{
			std::vector<int> keep;
			keep.reserve(n);
			for (int i = 0; i < n; ++i)
			{
				if (std::find(classes->begin(), classes->end(), cls_in[i]) != classes->end()) keep.push_back(i);
			}
			if (keep.empty()) return {};

			cv::Mat nb((int)keep.size(), 4, CV_32F);
			std::vector<float> nc; nc.reserve(keep.size());
			std::vector<int>   nl; nl.reserve(keep.size());
			orig_idx.clear(); orig_idx.reserve(keep.size());

			for (int k = 0; k < (int)keep.size(); ++k)
			{
				int i = keep[k];
				xyxy_in.row(i).copyTo(nb.row(k));
				nc.push_back(conf_in[i]);
				nl.push_back(cls_in[i]);
				orig_idx.push_back(i);
			}
			xyxy = std::move(nb);
			conf = std::move(nc);
			cls = std::move(nl);
			n = xyxy.rows;
		}
		else
		{
			// �޹��ˣ�ԭʼ����Ϊ [0..n-1]
			orig_idx.resize(n);
			std::iota(orig_idx.begin(), orig_idx.end(), 0);
		}

		// ---- 2) Ԥ�ü� top-k�������٣�
		if (n > max_nms)
		{
			std::vector<int> order(n);
			std::iota(order.begin(), order.end(), 0);
			std::partial_sort(order.begin(), order.begin() + max_nms, order.end(),
				[&](int i, int j) { return conf[i] > conf[j]; });

			cv::Mat nb(max_nms, 4, CV_32F);
			std::vector<float> nc; nc.reserve(max_nms);
			std::vector<int>   nl; nl.reserve(max_nms);
			std::vector<int>   no; no.reserve(max_nms);

			for (int k = 0; k < max_nms; ++k)
			{
				int i = order[k];
				xyxy.row(i).copyTo(nb.row(k));
				nc.push_back(conf[i]);
				nl.push_back(cls[i]);
				no.push_back(orig_idx[i]);
			}
			xyxy = std::move(nb);
			conf = std::move(nc);
			cls = std::move(nl);
			orig_idx = std::move(no);
			n = xyxy.rows;
		}

		// ---- 3) �����ƫ�ƣ����ƻ�ԭ xyxy_in��
		cv::Mat boxes_for_nms;
		xyxy.copyTo(boxes_for_nms);
		if (!agnostic)
		{
			for (int i = 0; i < n; ++i)
			{
				float c = (float)cls[i] * max_wh;
				float* b = boxes_for_nms.ptr<float>(i);
				b[0] += c; b[1] += c; b[2] += c; b[3] += c;
			}
		}

		// ---- 4) NMS���� extra_dedup��
		return nms_numpy_like_with_dedup(
			boxes_for_nms, conf, iou_thr, max_det,
			extra_dedup, overlap_thr, &orig_idx
		);
	}

	void scale_boxes(cv::Mat& boxes_xyxy, float r, const cv::Point& pad,
		int orig_w, int orig_h)
	{
		// boxes_xyxy: CV_32F, Nx4 (x1,y1,x2,y2)
		if (boxes_xyxy.empty()) return;
		CV_Assert(boxes_xyxy.type() == CV_32F && boxes_xyxy.cols == 4);

		boxes_xyxy.col(0) = (boxes_xyxy.col(0) - (float)pad.x) / r; // x1
		boxes_xyxy.col(2) = (boxes_xyxy.col(2) - (float)pad.x) / r; // x2
		boxes_xyxy.col(1) = (boxes_xyxy.col(1) - (float)pad.y) / r; // y1
		boxes_xyxy.col(3) = (boxes_xyxy.col(3) - (float)pad.y) / r; // y2

		const float W = static_cast<float>(orig_w);
		const float H = static_cast<float>(orig_h);
		for (int i = 0; i < boxes_xyxy.rows; ++i)
		{
			float& x1 = boxes_xyxy.at<float>(i, 0);
			float& y1 = boxes_xyxy.at<float>(i, 1);
			float& x2 = boxes_xyxy.at<float>(i, 2);
			float& y2 = boxes_xyxy.at<float>(i, 3);

			x1 = clamp(x1, 0.0f, W);
			x2 = clamp(x2, 0.0f, W);
			y1 = clamp(y1, 0.0f, H);
			y2 = clamp(y2, 0.0f, H);

			if (x2 < x1) x2 = x1;
			if (y2 < y1) y2 = y1;
		}
	}

	std::vector<itemX100CSFLocateInfo> postprocess_no_nms_cpp(const cv::Mat& pred,
		float conf_thr,
		float iou_thr,
		int   max_det,
		bool  agnostic,
		int   num_classes,
		bool  apply_sigmoid,
		std::vector<float> rs,
		std::vector<cv::Point> pads,
		bool  extra_dedup = false,
		const std::vector<int>* classes = nullptr)
	{

		if (pred.dims != 3 || pred.type() != CV_32F)
		{
			throw std::runtime_error("pred must be 3D CV_32F");
		}
		const int B = pred.size[0];
		const int A = pred.size[1];
		const int N = pred.size[2];


		const bool channels_first = (A == 5);
		if (!channels_first && N != 5)
			throw std::runtime_error("shape must be (B,5,N) or (B,N,5)");


		int C = 5 - 4;
		if (num_classes != C) num_classes = C;


		auto at3 = [&](int i, int j, int k)->float
		{
			int idx[3] = { i,j,k };
			return pred.at<float>(idx);
		};

		std::vector<itemX100CSFLocateInfo> out;
		out.reserve(B);

		for (int b = 0; b < B; ++b)
		{

			cv::Mat boxes_xywh(N, 4, CV_32F);
			cv::Mat scores(N, C, CV_32F);

			if (channels_first)
			{  // (B,6,N)
				for (int n = 0; n < N; ++n)
				{
					float* bx = boxes_xywh.ptr<float>(n);
					bx[0] = at3(b, 0, n); bx[1] = at3(b, 1, n);
					bx[2] = at3(b, 2, n); bx[3] = at3(b, 3, n);
					for (int c = 0; c < C; ++c) scores.at<float>(n, c) = at3(b, 4 + c, n);
				}
			}
			else
			{               // (B,N,6)
				for (int n = 0; n < N; ++n)
				{
					float* bx = boxes_xywh.ptr<float>(n);
					for (int k = 0; k < 4; ++k) { int idx[3] = { b,n,k }; bx[k] = pred.at<float>(idx); }
					for (int c = 0; c < C; ++c) { int idx[3] = { b,n,4 + c }; scores.at<float>(n, c) = pred.at<float>(idx); }
				}
			}

			if (apply_sigmoid)
			{
				for (int n = 0; n < N; ++n)
				{
					float* r = scores.ptr<float>(n);
					for (int c = 0; c < C; ++c) r[c] = 1.0f / (1.0f + std::exp(-r[c]));
				}
			}

			std::vector<float> conf(N);
			std::vector<int>   cls(N);
			for (int n = 0; n < N; ++n)
			{
				const float* r = scores.ptr<float>(n);
				int arg = 0; float mx = r[0];
				for (int c = 1; c < C; ++c) if (r[c] > mx) { mx = r[c]; arg = c; }
				conf[n] = mx; cls[n] = arg;
			}

			std::vector<int> keep_conf; keep_conf.reserve(N);
			for (int n = 0; n < N; ++n) if (conf[n] >= conf_thr) keep_conf.push_back(n);

			itemX100CSFLocateInfo dr;
			if (keep_conf.empty())
			{
				dr.boxes = cv::Mat(0, 4, CV_32F);
				out.emplace_back(std::move(dr));
				continue;
			}

			int M = (int)keep_conf.size();
			dr.boxes.create(M, 4, CV_32F);
			std::vector<float> cf; cf.reserve(M);
			std::vector<int>   cl; cl.reserve(M);

			for (int k = 0; k < M; ++k)
			{
				int n = keep_conf[k];
				const float* wh = boxes_xywh.ptr<float>(n);
				float x = wh[0], y = wh[1], w = wh[2], h = wh[3];
				float* bb = dr.boxes.ptr<float>(k);
				bb[0] = x - w * 0.5f; bb[1] = y - h * 0.5f; bb[2] = x + w * 0.5f; bb[3] = y + h * 0.5f; // xywh -> xyxy
				cf.push_back(conf[n]);
				cl.push_back(cls[n]);
			}
			auto keep = yolov_like_nms_cpp(dr.boxes, cf, cl,
				iou_thr, std::min(max_det, M),
				agnostic, 7680.0f, 30000, extra_dedup, 0.7, classes);

			std::vector<int> kk = keep;
			std::sort(kk.begin(), kk.end(), [&](int i, int j)
			{
				if (cf[i] != cf[j]) return cf[i] > cf[j];
				const float* bi = dr.boxes.ptr<float>(i);
				const float* bj = dr.boxes.ptr<float>(j);
				if (bi[0] != bj[0]) return bi[0] < bj[0];
				if (bi[1] != bj[1]) return bi[1] < bj[1];
				if (bi[2] != bj[2]) return bi[2] < bj[2];
				return bi[3] < bj[3];
			});

			itemX100CSFLocateInfo kept;
			kept.boxes.create((int)kk.size(), 4, CV_32F);
			kept.scores.reserve(kk.size());
			kept.labels.reserve(kk.size());
			for (int t = 0; t < (int)kk.size(); ++t)
			{
				int k = kk[t];
				dr.boxes.row(k).copyTo(kept.boxes.row(t));
				kept.scores.push_back(cf[k]);
				kept.labels.push_back(cl[k]);
			}
			scale_boxes(kept.boxes, rs[b], pads[b], 2448, 2048);
			out.emplace_back(std::move(kept));
		}
		return out;
	}
	std::vector<float> processInput(std::vector<cv::Mat> uImgs)
	{
		int len = BATCH * INPUTC * INPUTW * INPUTH;
        std::vector<float> chw(len);
		float* data = chw.data();

		for (int m = 0; m < BATCH; m++)
		{
			cv::Size new_shape(INPUTW, INPUTH);
			auto lb = letterbox(uImgs[m], new_shape, 0.5, cv::Scalar(114, 114, 114), true, false, true, true, 32, cv::INTER_LINEAR);
			rs.push_back(lb.r);
			pads.push_back(lb.pad);
			// BGR->RGB
			cv::Mat rgb;
			cv::cvtColor(lb.img, rgb, cv::COLOR_BGR2RGB);
			cv::Mat blob;
			rgb.convertTo(blob, CV_32FC3);
			blob = blob / 255.0;

			std::vector<cv::Mat> channels(INPUTC);
			for (int c = 0; c < INPUTC; ++c)
			{
				// 每个通道指向 chw 向量中对应的平面起始位置
				channels[c] = cv::Mat(INPUTH, INPUTW, CV_32FC1, data + m * INPUTC * INPUTH * INPUTW + c * INPUTH * INPUTW);
			}
			// 将 HWC 的 image 拆分并直接拷贝到 channels 指向的 chw 内存中
			cv::split(blob, channels);
		}
		return chw;
	}
	void postprocess(float* data, std::vector<itemX100CSFLocateInfo>& uOutValue)
	{
		cv::Mat pred_mat = makePredMatView(data, B, A, N);

		uOutValue = postprocess_no_nms_cpp(pred_mat, conf, iou, maxd, agn, 2, false, rs, pads, true, nullptr);
		// const auto& det = uOutValue[0];
		// std::cout << "det.boxes.rows " << det.boxes.rows << std::endl;
		// for (int r = 0; r < det.boxes.rows; ++r) {
		// 	float score = det.scores[r];

		// 	const float* b = det.boxes.ptr<float>(r); // [x1,y1,x2,y2]
		// 	int label = det.labels[r];

		// 	std::ostringstream box_ss;
		// 	box_ss << "["
		// 		<< std::fixed << std::setprecision(1) << b[0] << ", "
		// 		<< std::fixed << std::setprecision(1) << b[1] << ", "
		// 		<< std::fixed << std::setprecision(1) << b[2] << ", "
		// 		<< std::fixed << std::setprecision(1) << b[3] << "]";

		// 	std::cout << "name"
		// 		<< " -> Box: " << box_ss.str()
		// 		<< ", Score: " << std::fixed << std::setprecision(2) << score
		// 		<< ", Label: " << label
		// 		<< '\n';
		// }
	}
};
