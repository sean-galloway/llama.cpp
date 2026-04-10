// Flash-MoE CUDA Integration
// GPU-accelerated MoE computation with staging buffer

#ifndef FLASH_MOE_CUDA_H
#define FLASH_MOE_CUDA_H

#include "ggml.h"
#include "llama-staged-moe.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>

namespace llama {

// CUDA context for Flash-MoE
struct flash_moe_cuda_context {
    cublasHandle_t cublas_handle = nullptr;
    cudaStream_t stream = nullptr;
    int device_id = 0;
    bool initialized = false;

    // GPU staging buffers for active experts
    std::unique_ptr<float[]> gpu_gate_buffer;
    std::unique_ptr<float[]> gpu_up_buffer;
    std::unique_ptr<float[]> gpu_down_buffer;
    std::unique_ptr<float[]> gpu_cur_buffer;
    std::unique_ptr<float[]> gpu_output_buffer;
    std::unique_ptr<float[]> gpu_gate_up_buffer;

    size_t max_expert_size = 0;
    size_t max_embd = 0;
    size_t max_ff = 0;

    bool init(int device = 0, size_t n_embd = 7168, size_t n_ff = 2048);
    void free();
};

// GPU-accelerated matmul using cuBLAS
// C = alpha * op(A) @ op(B) + beta * C
void cuda_matmul(
    cublasHandle_t handle,
    const float* A, const float* B, float* C,
    int m, int n, int k,
    cublasOperation_t transA = CUBLAS_OP_N,
    cublasOperation_t transB = CUBLAS_OP_N,
    float alpha = 1.0f,
    float beta = 0.0f);

// Stage expert to GPU and compute
// Returns true if computation was done on GPU
bool flash_moe_expert_compute_cuda(
    flash_moe_cuda_context* cuda_ctx,
    const float* cur_token,           // Input token [n_embd]
    const float* gate_expert,         // Gate weights [n_ff, n_embd] or [n_embd, n_ff]
    const float* up_expert,           // Up weights [n_ff, n_embd] or [n_embd, n_ff]
    const float* down_expert,         // Down weights [n_embd, n_ff] or [n_ff, n_embd]
    float* output,                    // Output [n_embd]
    int64_t n_embd,
    int64_t n_ff,
    float expert_weight);

// Check if CUDA is available and working
bool flash_moe_cuda_available();

// Global CUDA context getter
flash_moe_cuda_context* get_flash_moe_cuda_context();

} // namespace llama

#endif // FLASH_MOE_CUDA_H
