#pragma once
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/process.hpp>
#include <thread>
#include <queue>
#include <string>
#include "common.hpp"

namespace bip = boost::interprocess;
namespace bp = boost::process;

std::string get_current_dir() {
    char path[PATH_MAX];
    return (getcwd(path, sizeof(path)) != nullptr)? std::string(path) : "";
}


class TaskWorkerWrapper {

public:

    TaskWorkerWrapper(int id) : worker_id_(id) {
        constexpr std::size_t SHM_OVERHEAD = 1024;
        constexpr std::size_t SHM_SIZE = sizeof(WorkerSharedBuffer) + SHM_OVERHEAD;
        buffname_ = "X40EnhanceWorkerSharedBuffer2_" + std::to_string(worker_id_);
        bip::shared_memory_object::remove(buffname_.c_str());
        shm_ = bip::managed_shared_memory(bip::create_only, buffname_.c_str(), SHM_SIZE);
        buffer_ = shm_.construct<WorkerSharedBuffer>(buffname_.c_str())();
        std::string exe_dir = get_current_dir();
        std::string consumer_path = exe_dir + "/algorithms/x40enhance/X40ImageEnhanceModelsWorker";
        try {
            printf("Starting process %d\n", id);
            // consumer_ = std::make_unique<bp::child>(consumer_path, buffname_, std::to_string(id));
            consumer_ = std::make_unique<bp::child>(consumer_path, buffname_, std::to_string(id),
                bp::std_out > ("test." + std::to_string(worker_id_) + ".log"));
        } catch (const bp::process_error &e) {
            std::cerr << "Error starting consumer process: " << e.what() << std::endl;
            throw;
        }
    }

    ~TaskWorkerWrapper() {
        if (consumer_ && consumer_->running()) {
            consumer_->terminate();
        }
        bip::shared_memory_object::remove(buffname_.c_str());
        std::cout << "Worker " << worker_id_ << " destroyed and shared memory removed." << std::endl;
    }

    WorkerSharedBuffer* get_buffer() {
        return buffer_;
    }

private:

    int                         worker_id_;
    std::unique_ptr<bp::child>  consumer_;
    std::string                 buffname_;
    bip::managed_shared_memory  shm_;
    WorkerSharedBuffer          *buffer_;

};
