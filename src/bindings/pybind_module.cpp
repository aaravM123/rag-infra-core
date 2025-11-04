#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// Cosine similarity between two float arrays
float cosine_similarity(const float* a, const float* b, size_t dim) {
    float dot = 0, norm_a = 0, norm_b = 0;
    #pragma omp parallel for reduction(+:dot,norm_a,norm_b)
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-9f);
}

// Top-K cosine similarity using NumPy arrays directly
py::array_t<int> search_topk_np(py::array_t<float> vectors,
                                py::array_t<float> query,
                                int k) {
    py::buffer_info vec_buf = vectors.request();
    py::buffer_info q_buf = query.request();

    auto* vec_data = static_cast<float*>(vec_buf.ptr);
    auto* q_data   = static_cast<float*>(q_buf.ptr);

    size_t num_vecs = vec_buf.shape[0];
    size_t dim      = vec_buf.shape[1];

    int actual_k = std::min(k, static_cast<int>(num_vecs));

    std::vector<std::pair<float,int>> scores(num_vecs);

    #pragma omp parallel for
    for (size_t i = 0; i < num_vecs; ++i) {
        float sim = cosine_similarity(&vec_data[i * dim], q_data, dim);
        scores[i] = {sim, (int)i};
    }

    std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
                      [](auto &a, auto &b){ return a.first > b.first; });

    py::array_t<int> topk(actual_k);
    auto r = topk.mutable_unchecked<1>();
    for (int i = 0; i < actual_k; ++i)
        r(i) = scores[i].second;

    return topk;
}

// Pybind11 module
PYBIND11_MODULE(vector_search, m) {
    m.def("search_topk_np", &search_topk_np, "Top-K cosine search (NumPy input)");
}
