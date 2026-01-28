#ifndef PUBLIC_TRT_HPP
#define PUBLIC_TRT_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <string>
#include <numeric>   // 必须包含，用于 std::accumulate
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include "gpu_info.h"

// 通用删除器：用于 std::unique_ptr 自动管理 TensorRT 对象的生命周期
struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const { if (obj) delete obj; }
};

/**
 * @brief TensorRT 10.x 通用基类
 * 支持：自动多输入输出识别、显存自动管理、基于名称的张量地址绑定
 */
class PublicTRT {
protected:
    std::unique_ptr<nvinfer1::IRuntime, TRTDeleter> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDeleter> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter> mContext;

    // mDeviceBindings: 存储所有张量（输入+输出）在 GPU 上的地址映射
    // Key: 模型中定义的 Tensor Name, Value: cudaMalloc 分配的 GPU 指针
    std::unordered_map<std::string, void*> mDeviceBindings;

    // mHostOutputs: 专门存储输出张量在 CPU 端的接收缓冲区
    // Key: 模型中定义的输出 Tensor Name, Value: 存储数据的 vector
    std::unordered_map<std::string, std::vector<float>> mHostOutputs;

	std::string getGPUName(int gpu_id) {
        // 假设 GPU_NAMES 是你全局定义的数组或 vector
        if (gpu_id < 0 || gpu_id >= GPU_COUNT) return "";
        return GPU_NAMES[gpu_id];
    }

	// 通用的路径选择器
    std::string selectEnginePath(int gpu_id, const std::string& modelName) {
        if (gpu_id < 0 || gpu_id >= GPU_COUNT) return "";
        
        std::string deviceName = GPU_NAMES[gpu_id];
        std::string folder = "";

        // 统一的路径判断逻辑
        if (deviceName.find("2080") != std::string::npos) folder = "2080";
        else if (deviceName.find("3080") != std::string::npos) folder = "3080";
        else if (deviceName.find("4070") != std::string::npos) folder = "4070";
        else if (deviceName.find("4090") != std::string::npos) folder = "4070"; // 兼容处理
        
        if (folder.empty()) return "";
        return "engines/" + folder + "/" + modelName + ".trt";
    }

public:
    virtual ~PublicTRT() {
        // 析构时自动释放所有在 mDeviceBindings 中分配的显存
        for (auto& pair : mDeviceBindings) {
            if (pair.second) {
                cudaFree(pair.second);
            }
        }
    }

    // 1. 初始化环境（插件、日志级别）
    void initTRT() {
        initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
        sample::gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kERROR);
    }

    // 2. 加载模型并自动绑定地址
    bool loadEngine(const std::string& enginePath) {
        std::ifstream engineFile(enginePath, std::ios::binary);
        if (!engineFile) {
            sample::gLogError << "无法打开 Engine 文件: " << enginePath << std::endl;
            return false;
        }

        engineFile.seekg(0, std::ios::end);
        size_t size = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        std::vector<char> engineData(size);
        engineFile.read(engineData.data(), size);

        mRuntime.reset(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
        mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), size));
        if (!mEngine) return false;

        mContext.reset(mEngine->createExecutionContext());

        /**
         * getNbIOTensors(): 获取模型中所有输入(Input)和输出(Output)张量的总数。
         * TensorRT 10.x 统一管理 IO 张量，不再区分 Index，而是通过 Name 访问。
         */
        for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
            // 获取张量名称（对应 ONNX 中的名称）
            auto name = mEngine->getIOTensorName(i);
            // 获取张量形状（Dims 对象，包含 nbDims 和 d[MAX_DIMS]）
            auto dims = mEngine->getTensorShape(name);
            // 获取 IO 模式：kINPUT（输入）或 kOUTPUT（输出）
            auto mode = mEngine->getTensorIOMode(name);

            /**
             * std::accumulate 计算元素总量：
             * dims.d: 形状数组的起始位置（如 [4, 3, 224, 224] 中的 4）
             * dims.d + dims.nbDims: 形状数组的结束位置
             * 1: 累乘初始值
             * std::multiplies<int64_t>(): 告诉 accumulate 做乘法操作
             */
            size_t count = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
            
            // 为该 Tensor 分配 GPU 显存
            void* devPtr = nullptr;
            cudaMalloc(&devPtr, count * sizeof(float));
            mDeviceBindings[name] = devPtr;

            // 如果该张量是输出，则在 CPU 上分配对应的 vector 空间用于接收推理结果
            if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                mHostOutputs[name].resize(count);
            }

            // 【核心关键】告知上下文：处理名为 'name' 的张量时，请使用 'devPtr' 指向的显存地址
            mContext->setTensorAddress(name, devPtr);
        }
        return true;
    }

    /**
     * @brief 执行推理
     * @param inputs 输入数据的映射。Key 是 Tensor 名，Value 是预处理后的数据向量。
     * 支持多输入：只需在 map 中放入多个 entry。
     */
    bool doInference(const std::unordered_map<std::string, std::vector<float>>& inputs) {
        if (!mContext) return false;

        // A. 将所有输入数据拷贝到模型对应的 GPU 地址
        for (const auto& [name, data] : inputs) {
            if (mDeviceBindings.count(name)) {
                // 根据输入名称精确找到其绑定的 GPU 地址进行拷贝
                cudaMemcpy(mDeviceBindings[name], data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
            } else {
                sample::gLogError << "错误：输入名 " << name << " 在模型中不存在！" << std::endl;
                return false;
            }
        }

        // B. 异步执行推理 (Enqueue V3)
        if (!mContext->enqueueV3(0)) return false;

        // C. 将所有标记为 kOUTPUT 的张量数据从 GPU 拷贝回 CPU
        for (auto& [name, hostVec] : mHostOutputs) {
            // 通过输出名称在 mDeviceBindings 映射表中找到计算完成的显存地址
            cudaMemcpy(hostVec.data(), mDeviceBindings[name], hostVec.size() * sizeof(float), cudaMemcpyDeviceToHost);
        }

        return true;
    }
};

#endif