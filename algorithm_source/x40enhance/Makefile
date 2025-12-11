# === Compiler and Python Settings ===
CXX = g++
PYTHON_VERSION = 3.10
PYTHON_LDFLAGS = $(shell python$(PYTHON_VERSION)-config --ldflags) -lpython$(PYTHON_VERSION)

PYBIND11_INCLUDES := $(shell python$(PYTHON_VERSION) -m pybind11 --includes)
PYTHON_SUFFIX := $(shell python$(PYTHON_VERSION)-config --extension-suffix)

# === Paths ===
SRC_DIR = src
INC_DIR = include
BIN_DIR = ../../algorithms/x40enhance

# === Source Files ===
PYBIND_SRC = $(SRC_DIR)/main.cpp
CONSUMER_SRC = $(SRC_DIR)/consumer.cpp

# # ====  c++test ===
# X40TEST_SRC = $(SRC_DIR)/test.cpp
# X40TEST_TARGET = $(BIN_DIR)/x40_task_test


PYBIND_TARGET = $(BIN_DIR)/X40ImageEnhanceModels$(PYTHON_SUFFIX)
CONSUMER_TARGET = $(BIN_DIR)/X40ImageEnhanceModelsWorker

# === cuda 设置 ===
CUDA_DIR = /usr/local/cuda
NVCC     = $(CUDA_DIR)/bin/nvcc
CUDA_INC = -I$(CUDA_DIR)/include
CUDA_LIB = -L$(CUDA_DIR)/lib64

# === 自动生成 gpu_info.h ===
GPU_INFO_H := $(INC_DIR)/gpu_info.h
QUERY_GPU  := $(BIN_DIR)/query_gpu


# === OpenCV Config ===
OPENCV_DIR = ../../opencv4
OPENCV_INC = -I$(OPENCV_DIR)/include/opencv4
OPENCV_LIB = -L$(OPENCV_DIR)/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn -lopencv_highgui

# === Boost ===
BOOST_LIBS = -lboost_system -lboost_filesystem -lboost_thread -lboost_program_options

# === TensorRT 配置 ===
TENSORRT_DIR  = ../../TensorRT-8.6.1.6
TENSORRT_INC  = -I$(TENSORRT_DIR)/include
TENSORRT_COMMON = -I$(TENSORRT_DIR)/samples/common
TENSORRT_COMMON_SRC = \
	$(TENSORRT_DIR)/samples/common/logger.cpp 

TENSORRT_LIB  = -L$(TENSORRT_DIR)/lib
TENSORRT_LIBS = -lnvinfer -lnvonnxparser -lnvparsers -lnvinfer_plugin

# === RPATH ===
RPATH := -Wl,-rpath,$(realpath $(OPENCV_DIR)/lib):$(realpath $(TENSORRT_DIR)/lib)

# === Flags ===
CXXFLAGS = -O3 -Wall -std=c++17 -fPIC -I$(INC_DIR) $(TENSORRT_INC) $(TENSORRT_COMMON) $(CUDA_INC) $(PYBIND11_INCLUDES)
LDFLAGS = $(RPATH) $(BOOST_LIBS) $(OPENCV_LIBS) $(TENSORRT_LIBS) $(CUDA_LIB)  $(PYTHON_LDFLAGS) -lcudart

CXXFLAGS += -Wno-deprecated-declarations

# === Build Rules ===

all: $(BIN_DIR) $(PYBIND_TARGET) $(GPU_INFO_H) $(CONSUMER_TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# 编译 query_gpu.cu 到指定目录
$(QUERY_GPU): $(SRC_DIR)/query_gpu.cu | $(BIN_DIR)
	$(NVCC) -o $@ $< $(CUDA_INC) $(CUDA_LIB)

# 生成 gpu_info.h
$(GPU_INFO_H): $(QUERY_GPU)
	$< | awk 'BEGIN{print "static const std::string GPU_NAMES[] = {"} \
	{id=$$1; $$1=""; sub(/^ /,""); print "    \""$$0"\","} \
	END{print "};\n#define GPU_COUNT " NR}' > $(GPU_INFO_H)

# $(X40TASK_TARGET): $(X40TASK_SRC)
# 	$(CXX) $(CXXFLAGS) $(OPENCV_INC) -o $@ $^ \
# 	    $(OPENCV_LIB) $(TENSORRT_LIB) $(LDFLAGS)

# === Pybind11 Python Module ===
$(PYBIND_TARGET): $(PYBIND_SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(OPENCV_INC) -shared $^ -o $@ \
	    $(OPENCV_LIB) $(TENSORRT_LIB) $(LDFLAGS)


# === Standalone Consumer Binary ===
$(CONSUMER_TARGET): $(CONSUMER_SRC) $(TENSORRT_COMMON_SRC)
	$(CXX) $(CXXFLAGS) $(OPENCV_INC) -o $@ $^ \
	    $(OPENCV_LIB) $(TENSORRT_LIB) $(LDFLAGS)



clean:
	rm -f $(PYBIND_TARGET) $(CONSUMER_TARGET) $(X40TASK_TARGET) $(QUERY_GPU)