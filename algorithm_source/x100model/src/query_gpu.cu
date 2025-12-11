#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << i << " " << prop.name << std::endl;
    }
    return 0;
}
