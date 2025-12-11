#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dispatcher.hpp"

namespace py = pybind11;

PYBIND11_MODULE(X100ImageModels, m) {
    py::class_<X100ImageModels>(m, "X100ImageModels")
        .def(py::init<int>(), py::arg("num_workers"))
        .def("enqueue_task", &X100ImageModels::enqueue_task, py::arg("image"), py::arg("task_type"), "Enqueue a task with an image for processing")
        .def("synchronize", &X100ImageModels::synchronize, "Finalize the task queue and ensure all tasks are processed")
        .def("get_result", &X100ImageModels::get_result, py::arg("task_id"), "Get the result of a processed task by its ID")
    ;
}