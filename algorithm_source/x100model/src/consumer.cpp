#include <iostream>
#include <string>
#include <thread>
#include <boost/interprocess/managed_shared_memory.hpp>
#include "common.hpp"
#include "process.hpp"
// #include "test.hpp"

namespace bip = boost::interprocess;

int main(int argc, char *argv[]) {

    std::string buffname(argv[1]);
    std::cout << "Consumer started with buffer name: " << buffname << std::endl;
    bip::managed_shared_memory shm(bip::open_only, buffname.c_str());
    WorkerSharedBuffer *buffer = shm.find<WorkerSharedBuffer>(buffname.c_str()).first;

    // int device_count = 0;
    // cudaError_t err = cudaGetDeviceCount(&device_count);
    // if (err != cudaSuccess) {
    //     std::cerr << "cudaGetDeviceCount failed: "
    //               << cudaGetErrorString(err) << std::endl;
    //     return -1;
    // }
    int gpu_id = std::stoi(argv[2]) % GPU_COUNT; 
    // int gpu_id = 0;

    std::cout << "显卡 " << gpu_id << "开启子进程" << std::endl;

    if (!buffer) {
        std::cerr << "Failed to find shared buffer: " << buffname << std::endl;
        return -1;
    }
    setenv("CUDA_VISIBLE_DEVICES", std::to_string(gpu_id).c_str(), 1);
    cudaSetDevice(gpu_id); 
    X100Main *x100 = new X100Main(std::stoi(argv[2]), gpu_id);

    while (true) {
        int slot_id = buffer->wait_next_ready_block();
        WorkerDataBlock &block = buffer->blocks[slot_id];
        {
            block.task_status = COMPUTING;
            buffer->cv_result_ready_.notify_all();
            process_batch(buffer, block, slot_id, x100, gpu_id);
        }
    }
    return 0;    
}
