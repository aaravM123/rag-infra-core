#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>

float cosine_similarity(cons std::vector<float>& a, const std::vector<float>&b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    #pragma omp parallel for reduction(+:dot, norm_a, norm_b)
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-9f);
}

float l2_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}


std::vector<int> search_topk(const std::vector<std::vector<float>>& vectors, const std::vector<float>&query, int k) {
    std::vector<std::pair<float,int>> scores;
    scores.reserve(vectors.size());
    
    #pragma omp parallel for {
        std::vector<std::pair<float,int>> local;
        #pragma omp for nowait
        for (size_t i = 0; i < vectors.size(); ++i) {
            local.emplace_back(cosine_similarity(vectors[i], query), i);
        #pragma omp critical
        scores.insert(scores.end(), local.begin(), local.end());
        }
    }

    std::partial_sort(scores.begin(), scores.begin()+k, scores.end(), [](auto& a, auto& b) { return a.first > b.first; });
    
    std::vector<int> topk;
    for (int i = 0; i < k; ++i) {
        topk.push_back(scores[i].second);
        return topk;
    }   