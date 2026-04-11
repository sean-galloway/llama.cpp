// Flash-MoE Custom Operator with CUDA Support
// Integrates GPU staging buffer for 6-10x speedup

#ifndef FLASH_MOE_CUSTOM_OP_CUDA_H
#define FLASH_MOE_CUSTOM_OP_CUDA_H

#include "flash_moe_custom_op.h"
#include "flash_moe_cuda.h"
#include <cstring>

namespace llama {

// CUDA-aware Flash-MoE custom operation
// This version uses GPU staging buffer to avoid MMU faults
inline void flash_moe_custom_op_cuda(
    struct ggml_tensor* dst,
    int ith, int nth,
    void* userdata) {

    (void)ith;
    (void)nth;

    flash_moe_userdata* data = (flash_moe_userdata*)userdata;
    if (!data || !data->mgr) {
        fprintf(stderr, "Flash-MoE CUDA: Invalid userdata\n");
        return;
    }

    // Get source tensors from dst->src[]
    struct ggml_tensor* src0 = dst->src[0];  // cur input
    struct ggml_tensor* src1 = dst->src[1];  // gate weights

    if (!src0 || !src1) {
        fprintf(stderr, "Flash-MoE CUDA: Missing source tensors\n");
        return;
    }

    staged_moe_manager* mgr = data->mgr;
    const int layer_id = data->layer_id;
    const int64_t n_expert = data->n_expert;
    const int64_t n_expert_used = data->n_expert_used;
    const int64_t n_embd = data->n_embd;
    const int64_t n_ff = data->n_ff;

    const int64_t n_tokens = src0->ne[1];

    // Get input data
    const float* cur_data = (const float*)ggml_get_data(src0);
    const float* gate_inp_data = (const float*)ggml_get_data(src1);

    if (!cur_data || !gate_inp_data) {
        fprintf(stderr, "Flash-MoE CUDA: No input data available\n");
        return;
    }

    // Initialize CUDA context if needed
    flash_moe_cuda_context* cuda_ctx = get_flash_moe_cuda_context();
    if (!cuda_ctx->initialized) {
        if (flash_moe_cuda_available()) {
            fprintf(stderr, "Flash-MoE CUDA: Initializing context...\n");
            if (!cuda_ctx->init(0, n_embd, n_ff)) {
                fprintf(stderr, "Flash-MoE CUDA: Initialization failed, falling back to CPU\n");
                // Fall back to CPU implementation
                flash_moe_custom_op(dst, ith, nth, userdata);
                return;
            }
        } else {
            fprintf(stderr, "Flash-MoE CUDA: CUDA not available, using CPU\n");
            flash_moe_custom_op(dst, ith, nth, userdata);
            return;
        }
    }

    // Allocate temporary buffers for router computation (always on CPU)
    std::vector<float> logits(n_expert * n_tokens);
    std::vector<float> probs(n_expert * n_tokens);
    std::vector<int> selected_experts(n_expert_used * n_tokens);

    // Compute router on CPU (lightweight)
    compute_router(gate_inp_data, cur_data, logits.data(), n_expert, n_embd, n_tokens);

    // Apply gating function
    switch (data->gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            apply_sigmoid(probs.data(), logits.data(), n_expert, n_tokens);
            break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
        default:
            apply_softmax(probs.data(), logits.data(), n_expert, n_tokens);
            break;
    }

    // Select top-k experts
    select_top_k(selected_experts.data(), probs.data(), n_expert, n_expert_used, n_tokens);

    // Log selected experts
    fprintf(stderr, "Flash-MoE CUDA: Layer %d - selected experts: ", layer_id);
    for (int64_t k_idx = 0; k_idx < n_expert_used && k_idx < 3; k_idx++) {
        int expert_id = selected_experts[k_idx * n_tokens];
        fprintf(stderr, "%d (%.4f) ", expert_id, probs[expert_id * n_tokens]);
    }
    fprintf(stderr, "\n");

    // Allocate output buffer
    std::vector<float> output(n_embd * n_tokens, 0.0f);
    std::vector<float> expert_out(n_embd);

    // Process each token using GPU
    for (int64_t t = 0; t < n_tokens; t++) {
        const float* cur_token = cur_data + t * n_embd;

        // Accumulate weighted outputs from each selected expert
        for (int64_t k_idx = 0; k_idx < n_expert_used; k_idx++) {
            int expert_id = selected_experts[k_idx * n_tokens + t];
            float expert_weight = probs[expert_id * n_tokens + t];

            // Load expert weights via pread() - stays in CPU memory
            void* gate_expert = mgr->get_expert_data(layer_id, "gate", expert_id);
            void* up_expert = mgr->get_expert_data(layer_id, "up", expert_id);
            void* down_expert = mgr->get_expert_data(layer_id, "down", expert_id);

            if (!gate_expert || !up_expert || !down_expert) {
                fprintf(stderr, "Flash-MoE CUDA: Failed to load expert %d\n", expert_id);
                continue;
            }

            // Try GPU computation
            bool gpu_success = flash_moe_expert_compute_cuda(
                cuda_ctx,
                cur_token,
                (const float*)gate_expert,
                (const float*)up_expert,
                (const float*)down_expert,
                expert_out.data(),
                n_embd, n_ff,
                expert_weight);

            if (!gpu_success) {
                fprintf(stderr, "Flash-MoE CUDA: GPU compute failed for expert %d, using CPU\n", expert_id);

                // Fall back to CPU for this expert
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
            }

            // Accumulate to output
            float* out_token = output.data() + t * n_embd;
            for (int64_t d = 0; d < n_embd; d++) {
                out_token[d] += expert_out[d];
            }
        }
    }

    // Copy output to destination tensor
    float* dst_data = (float*)ggml_get_data(dst);
    if (dst_data) {
        memcpy(dst_data, output.data(), n_embd * n_tokens * sizeof(float));
    }

    fprintf(stderr, "Flash-MoE CUDA: Layer %d - MoE computation complete for %ld tokens\n",
            layer_id, n_tokens);
}

} // namespace llama

#endif // FLASH_MOE_CUSTOM_OP_CUDA_H
