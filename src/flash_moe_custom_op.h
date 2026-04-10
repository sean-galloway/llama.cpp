// Flash-MoE Custom Operator Interface
// CPU implementation of Flash-MoE computation

#ifndef FLASH_MOE_CUSTOM_OP_H
#define FLASH_MOE_CUSTOM_OP_H

#include "ggml.h"
#include "llama-staged-moe.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace llama {

// Forward declarations from llama-model
enum llama_expert_gating_func_type;

// User data passed to custom operator
struct flash_moe_userdata {
    staged_moe_manager* mgr = nullptr;
    int layer_id = 0;
    int64_t n_expert = 0;
    int64_t n_expert_used = 0;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    llama_expert_gating_func_type gating_op;
};

// CPU matrix multiplication: C = A^T @ B
// A: [m, k], B: [n, k], C: [m, n]
inline void matmul_t(const float* A, const float* B, float* C, int64_t m, int64_t n, int64_t k) {
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t l = 0; l < k; l++) {
                sum += A[i * k + l] * B[j * k + l];
            }
            C[i * n + j] = sum;
        }
    }
}

// Compute router logits
// cur: [n_embd, n_tokens], gate_inp: [n_expert, n_embd]
// logits: [n_expert, n_tokens]
inline void compute_router(
    const float* gate_inp,
    const float* cur,
    float* logits,
    int64_t n_expert,
    int64_t n_embd,
    int64_t n_tokens) {

    // logits = gate_inp @ cur
    // [n_expert, n_tokens] = [n_expert, n_embd] @ [n_embd, n_tokens]
    for (int64_t e = 0; e < n_expert; e++) {
        for (int64_t t = 0; t < n_tokens; t++) {
            float sum = 0.0f;
            for (int64_t d = 0; d < n_embd; d++) {
                sum += gate_inp[e * n_embd + d] * cur[t * n_embd + d];
            }
            logits[e * n_tokens + t] = sum;
        }
    }
}

// Apply softmax to logits
inline void apply_softmax(float* probs, const float* logits, int64_t n_expert, int64_t n_tokens) {
    for (int64_t t = 0; t < n_tokens; t++) {
        // Find max for numerical stability
        float max_logit = logits[t];
        for (int64_t e = 1; e < n_expert; e++) {
            max_logit = std::max(max_logit, logits[e * n_tokens + t]);
        }

        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int64_t e = 0; e < n_expert; e++) {
            float exp_val = std::exp(logits[e * n_tokens + t] - max_logit);
            probs[e * n_tokens + t] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize
        for (int64_t e = 0; e < n_expert; e++) {
            probs[e * n_tokens + t] /= sum_exp;
        }
    }
}

// Apply sigmoid to logits
inline void apply_sigmoid(float* probs, const float* logits, int64_t n_expert, int64_t n_tokens) {
    for (int64_t e = 0; e < n_expert; e++) {
        for (int64_t t = 0; t < n_tokens; t++) {
            probs[e * n_tokens + t] = 1.0f / (1.0f + std::exp(-logits[e * n_tokens + t]));
        }
    }
}

// Select top-k experts
inline void select_top_k(int* selected, const float* probs, int64_t n_expert, int64_t k, int64_t n_tokens) {
    for (int64_t t = 0; t < n_tokens; t++) {
        // Simple selection: track top k
        std::vector<std::pair<float, int>> expert_scores;
        for (int64_t e = 0; e < n_expert; e++) {
            expert_scores.push_back({probs[e * n_tokens + t], (int)e});
        }

        // Partial sort to get top k
        std::partial_sort(expert_scores.begin(), expert_scores.begin() + k, expert_scores.end(),
            std::greater<std::pair<float, int>>());

        for (int64_t i = 0; i < k; i++) {
            selected[i * n_tokens + t] = expert_scores[i].second;
        }
    }
}

// CPU implementation of Flash-MoE custom operator
inline void flash_moe_custom_op(
    struct ggml_tensor* dst,
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1,
    int ith, int nth,
    void* userdata) {

    (void)ith;
    (void)nth;

    flash_moe_userdata* data = (flash_moe_userdata*)userdata;
    if (!data || !data->mgr) {
        fprintf(stderr, "Flash-MoE: Invalid userdata\n");
        return;
    }

    staged_moe_manager* mgr = data->mgr;
    const int layer_id = data->layer_id;
    const int64_t n_expert = data->n_expert;
    const int64_t n_expert_used = data->n_expert_used;
    const int64_t n_embd = data->n_embd;
    const int64_t n_ff = data->n_ff;

    const int64_t n_tokens = src0->ne[1];

    const float* cur_data = (const float*)ggml_get_data(src0);
    const float* gate_inp_data = (const float*)ggml_get_data(src1);

    if (!cur_data || !gate_inp_data) {
        fprintf(stderr, "Flash-MoE: No input data available\n");
        return;
    }

    // Allocate temporary buffers
    std::vector<float> logits(n_expert * n_tokens);
    std::vector<float> probs(n_expert * n_tokens);
    std::vector<int> selected_experts(n_expert_used * n_tokens);

    // Compute router
    compute_router(gate_inp_data, cur_data, logits.data(), n_expert, n_embd, n_tokens);

    // Apply gating
    switch (data->gating_op) {
        case 1: // SIGMOID
            apply_sigmoid(probs.data(), logits.data(), n_expert, n_tokens);
            break;
        case 0: // SOFTMAX
        default:
            apply_softmax(probs.data(), logits.data(), n_expert, n_tokens);
            break;
    }

    // Select top-k
    select_top_k(selected_experts.data(), probs.data(), n_expert, n_expert_used, n_tokens);

    // Allocate output buffer
    std::vector<float> output(n_embd * n_tokens, 0.0f);
    std::vector<float> expert_out(n_embd);

    // Process each token
    for (int64_t t = 0; t < n_tokens; t++) {
        const float* cur_token = cur_data + t * n_embd;

        for (int64_t k_idx = 0; k_idx < n_expert_used; k_idx++) {
            int expert_id = selected_experts[k_idx * n_tokens + t];
            float expert_weight = probs[expert_id * n_tokens + t];

            // Load expert weights
            void* gate_expert = mgr->get_expert_data(layer_id, "gate", expert_id);
            void* up_expert = mgr->get_expert_data(layer_id, "up", expert_id);
            void* down_expert = mgr->get_expert_data(layer_id, "down", expert_id);

            if (!gate_expert || !up_expert || !down_expert) {
                fprintf(stderr, "Flash-MoE: Failed to load expert %d\n", expert_id);
                continue;
            }

            // CPU computation
            std::vector<float> gate_buf(n_ff);
            std::vector<float> up_buf(n_ff);
            std::vector<float> gate_up_buf(n_ff);

            matmul_t(cur_token, (const float*)gate_expert, gate_buf.data(), 1, n_ff, n_embd);
            matmul_t(cur_token, (const float*)up_expert, up_buf.data(), 1, n_ff, n_embd);

            // Apply SILU
            for (int64_t i = 0; i < n_ff; i++) {
                float sigmoid = 1.0f / (1.0f + std::exp(-gate_buf[i]));
                gate_buf[i] = gate_buf[i] * sigmoid;
            }

            // Element-wise multiply
            for (int64_t i = 0; i < n_ff; i++) {
                gate_up_buf[i] = gate_buf[i] * up_buf[i];
            }

            // Compute output
            matmul_t(gate_up_buf.data(), (const float*)down_expert, expert_out.data(), 1, n_embd, n_ff);

            // Apply weight
            for (int64_t i = 0; i < n_embd; i++) {
                expert_out[i] *= expert_weight;
            }

            // Accumulate
            float* out_token = output.data() + t * n_embd;
            for (int64_t i = 0; i < n_embd; i++) {
                out_token[i] += expert_out[i];
            }
        }
    }

    // Copy output
    float* dst_data = (float*)ggml_get_data(dst);
    if (dst_data) {
        std::memcpy(dst_data, output.data(), n_embd * n_tokens * sizeof(float));
    }
}

} // namespace llama

#endif // FLASH_MOE_CUSTOM_OP_H
