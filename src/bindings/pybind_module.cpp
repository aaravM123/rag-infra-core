#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

float cosine_similarity(const std::vector<float>&, const std::vector<float>&);
float l2_distance(const std::vector<float>&, const std::vector<float>&);
std::vector<int> search_topk(const std::vector<std::vector<float>>&, const std::vector<float>&, int);

PYBIND11_MODULE(vector_search, m) {
    m.doc() = "C++ vector similarity kernels";
    m.def("cosine_similarity", &cosine_similarity);
    m.def("l2_distance", &l2_distance);
    m.def("search_topk", &search_topk);
}

