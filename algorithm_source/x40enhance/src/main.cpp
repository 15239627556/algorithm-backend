#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dispatcher.hpp"

namespace py = pybind11;

PYBIND11_MODULE(X40ImageEnhanceModels, m) {
    py::class_<X40ImageEnhanceModels>(m, "X40ImageEnhanceModels")
        .def(py::init<int>(), py::arg("num_workers"))
        .def("enqueue_task", &X40ImageEnhanceModels::enqueue_task, py::arg("image"), "Enqueue a task with an image for processing")
        .def("synchronize", &X40ImageEnhanceModels::synchronize, "Finalize the task queue and ensure all tasks are processed")
        .def("get_result", &X40ImageEnhanceModels::get_result, py::arg("task_id"), "Get the result of a processed task by its ID")
    ;
}