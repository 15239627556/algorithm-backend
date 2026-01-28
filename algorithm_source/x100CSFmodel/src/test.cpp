// #include "x40Task3.hpp"
#include <chrono>
#include <thread>
using namespace std;

int main()
{
    // X40Main *uInfo = new X40Main();
    // // std::cout << uInfo << std::endl;
    // cv::Mat im1 = cv::imread("/home/zwxk/pythonProjects/ModelCalculation/AlgorithmService/modelCalculation/images/1.jpg");
    // std::vector<cv::Mat> images;
    // for(int i = 0; i < 4; i ++)
    // {
    //     images.push_back(im1);
    // }
    // auto now = std::chrono::system_clock::now();                     // 当前时间点
    // std::time_t now_time = std::chrono::system_clock::to_time_t(now); // 转为 time_t
    // std::tm tm_time = *std::localtime(&now_time);                    // 转为本地时间结构体

    // std::ostringstream oss;
    // oss << std::put_time(&tm_time, "%Y-%m-%d %H:%M:%S");
    // std::cout << "当前时间: " << oss.str() << std::endl;
    // for(int i = 0; i < 50; i++)
    // {
    //     uInfo->add_x40_task(images);
    //     std::cout << "添加第 "  << i << "个任务" << std::endl;
    //     // std::this_thread::sleep_for(std::chrono::seconds(1)); 
    // }
    // // auto end = std::chrono::high_resolution_clock::now();

    // // 计算耗时（毫秒）
    // // std::chrono::duration<double, std::milli> duration = end - start;
    // // std::cout << "耗时: " << duration.count() << " ms" << std::endl;
    // std::this_thread::sleep_for(std::chrono::seconds(120));           // 睡 n 秒
    // delete uInfo;
    return 0;
}