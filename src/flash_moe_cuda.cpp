// Flash-MoE CUDA Implementation
// GPU-accelerated MoE computation with staging buffer

#include "flash_moe_cuda.h"
#include <cstdio>
#include <cstring>

namespace llama {

// Global CUDA context
static flash_moe_cuda_context g_flash_moe_cuda_ctx;

flash_moe_cuda_context* get_flash_moe_cuda_context() {
    return &g_flash_moe_cuda_ctx;
}

bool flash_moe_cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }

    // Check if we can allocate a small test buffer
    float* test_buf = nullptr;
    err = cudaMalloc(&test_buf, 1024);
    if (err != cudaSuccess) {
        return false;
    }
    cudaFree(test_buf);

    return true;
}

bool flash_moe_cuda_context::init(int device, size_t n_embd, size_t n_ff) {
    if (initialized) {
        return true;
    }

    device_id = device;
    max_embd = n_embd;
    max_ff = n_ff;

    // Set device
    cudaError_t cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "Flash-MoE CUDA: Failed to set device %d: %s\n",
                device_id, cudaGetErrorString(cuda_err));
        return false;
    }

    // Create cuBLAS handle
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Flash-MoE CUDA: Failed to create cuBLAS handle\n");
        return false;
    }

    // Create CUDA stream
    cuda_err = cudaStreamCreate(&stream);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "Flash-MoE CUDA: Failed to create stream: %s\n",
                cudaGetErrorString(cuda_err));
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
        return false;
    }

    // Set stream for cuBLAS
    cublasSetStream(cublas_handle, stream);

    // Calculate buffer sizes
    // For K=8 experts, we need staging buffers for:
    // - One expert's gate weights: n_ff * n_embd
    // - One expert's up weights: n_ff * n_embd
    // - One expert's down weights: n_embd * n_ff
    // - Current token: n_embd
    // - Output: n_embd
    // - Intermediate gate_up: n_ff

    max_expert_size = std::max(n_ff * n_embd, n_embd * n_ff);

    // Allocate GPU buffers
    size_t gate_size = n_ff * n_embd;
    size_t up_size = n_ff * n_embd;
    size_t down_size = n_embd * n_ff;
    size_t cur_size = n_embd;
    size_t out_size = n_embd;
    size_t gate_up_size = n_ff;

    // Use cudaMalloc for device memory
    float* gpu_gate = nullptr;
    float* gpu_up = nullptr;
    float* gpu_down = nullptr;
    float* gpu_cur = nullptr;
    float* gpu_out = nullptr;
    float* gpu_gate_up = nullptr;

    cuda_err = cudaMalloc(&gpu_gate, gate_size * sizeof(float));
    if (cuda_err != cudaSuccess) goto cleanup;

    cuda_err = cudaMalloc(&gpu_up, up_size * sizeof(float));
    if (cuda_err != cudaSuccess) goto cleanup;

    cuda_err = cudaMalloc(&gpu_down, down_size * sizeof(float));
    if (cuda_err != cudaSuccess) goto cleanup;

    cuda_err = cudaMalloc(&gpu_cur, cur_size * sizeof(float));
    if (cuda_err != cudaSuccess) goto cleanup;

    cuda_err = cudaMalloc(&gpu_out, out_size * sizeof(float));
    if (cuda_err != cudaSuccess) goto cleanup;

    cuda_err = cudaMalloc(&gpu_gate_up, gate_up_size * sizeof(float));
    if (cuda_err != cudaSuccess) goto cleanup;

    // Wrap raw pointers in unique_ptr with custom deleter
    gpu_gate_buffer.reset(gpu_gate);
    gpu_up_buffer.reset(gpu_up);
    gpu_down_buffer.reset(gpu_down);
    gpu_cur_buffer.reset(gpu_cur);
    gpu_output_buffer.reset(gpu_out);
    gpu_gate_up_buffer.reset(gpu_gate_up);

    initialized = true;
    fprintf(stderr, "Flash-MoE CUDA: Initialized successfully\n");
    fprintf(stderr, "  Device: %d\n", device_id);
    fprintf(stderr, "  Staging buffers: %.2f MB total\n",
            (gate_size + up_size + down_size + cur_size + out_size + gate_up_size) * sizeof(float) / (1024.0 * 1024.0));

    return true;

cleanup:
    fprintf(stderr, "Flash-MoE CUDA: Failed to allocate GPU memory: %s\n",
            cudaGetErrorString(cuda_err));
    if (gpu_gate) cudaFree(gpu_gate);
    if (gpu_up) cudaFree(gpu_up);
    if (gpu_down) cudaFree(gpu_down);
    if (gpu_cur) cudaFree(gpu_cur);
    if (gpu_out) cudaFree(gpu_out);
    if (gpu_gate_up) cudaFree(gpu_gate_up);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    cublas_handle = nullptr;
    stream = nullptr;
    return false;
}

void flash_moe_cuda_context::free() {
    if (!initialized) return;

    // Synchronize stream
    if (stream) {
        cudaStreamSynchronize(stream);
    }

    // Free cuBLAS handle
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }

    // Free GPU buffers (unique_ptr will call cudaFree via custom deleter if we set one)
    if (gpu_gate_buffer) { cudaFree(gpu_gate_buffer.get()); gpu_gate_buffer.release(); }
    if (gpu_up_buffer) { cudaFree(gpu_up_buffer.get()); gpu_up_buffer.release(); }
    if (gpu_down_buffer) { cudaFree(gpu_down_buffer.get()); gpu_down_buffer.release(); }
    if (gpu_cur_buffer) { cudaFree(gpu_cur_buffer.get()); gpu_cur_buffer.release(); }
    if (gpu_output_buffer) { cudaFree(gpu_output_buffer.get()); gpu_output_buffer.release(); }
    if (gpu_gate_up_buffer) { cudaFree(gpu_gate_up_buffer.get()); gpu_gate_up_buffer.release(); }

    // Destroy stream
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }

    initialized = false;
    fprintf(stderr, "Flash-MoE CUDA: Freed resources\n");
}

// GPU-accelerated matmul using cuBLAS
void cuda_matmul(
    cublasHandle_t handle,
    const float* A, const float* B, float* C,
    int m, int n, int k,
    cublasOperation_t transA,
    cublasOperation_t transB,
    float alpha,
    float beta) {

    // cuBLAS is column-major, but our data is row-major
    // For row-major C = A @ B, we compute C^T = B^T @ A^T in column-major

    int lda = (transA == CUBLAS_OP_N) ? k : m;
    int ldb = (transB == CUBLAS_OP_N) ? n : k;
    int ldc = n;

    cublasStatus_t status = cublasSgemm(
        handle,
        transB, transA,  // Note: swapped for row-major
        n, m, k,
        &alpha,
        B, ldb,
        A, lda,
        &beta,
        C, ldc
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Flash-MoE CUDA: cuBLAS sgemm failed: %d\n", status);
    }
}

// Stage expert to GPU and compute
bool flash_moe_expert_compute_cuda(
    flash_moe_cuda_context* cuda_ctx,
    const float* cur_token,
    const float* gate_expert,
    const float* up_expert,
    const float* down_expert,
    float* output,
    int64_t n_embd,
    int64_t n_ff,
    float expert_weight) {

    if (!cuda_ctx || !cuda_ctx->initialized) {
        return false;
    }

    cudaError_t cuda_err;
    cublasHandle_t handle = cuda_ctx->cublas_handle;
    cudaStream_t stream = cuda_ctx->stream;

    // Copy cur_token to GPU
    cuda_err = cudaMemcpyAsync(
        cuda_ctx->gpu_cur_buffer.get(), cur_token,
        n_embd * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) return false;

    // Copy gate_expert to GPU [n_ff, n_embd]
    cuda_err = cudaMemcpyAsync(
        cuda_ctx->gpu_gate_buffer.get(), gate_expert,
        n_ff * n_embd * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) return false;

    // Copy up_expert to GPU [n_ff, n_embd]
    cuda_err = cudaMemcpyAsync(
        cuda_ctx->gpu_up_buffer.get(), up_expert,
        n_ff * n_embd * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) return false;

    // Copy down_expert to GPU [n_embd, n_ff]
    cuda_err = cudaMemcpyAsync(
        cuda_ctx->gpu_down_buffer.get(), down_expert,
        n_embd * n_ff * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) return false;

    // Compute gate = cur @ gate_expert.T
    // cur: [1, n_embd], gate_expert: [n_ff, n_embd]
    // gate_expert is stored as [n_ff, n_embd] = [rows, cols]
    // For matmul: gate = cur @ gate_expert.T
    // gate_expert.T: [n_embd, n_ff]
    // cur @ gate_expert.T: [1, n_embd] @ [n_embd, n_ff] = [1, n_ff]
    cuda_matmul(handle,
                cuda_ctx->gpu_cur_buffer.get(),
                cuda_ctx->gpu_gate_buffer.get(),
                cuda_ctx->gpu_gate_up_buffer.get(),  // Use as temp
                1, n_ff, n_embd,
                CUBLAS_OP_N, CUBLAS_OP_T);  // B is transposed

    // Compute up = cur @ up_expert.T -> store in gpu_up_buffer
    cuda_matmul(handle,
                cuda_ctx->gpu_cur_buffer.get(),
                cuda_ctx->gpu_up_buffer.get(),
                cuda_ctx->gpu_up_buffer.get(),  // Overwrite
                1, n_ff, n_embd,
                CUBLAS_OP_N, CUBLAS_OP_T);

    // Synchronize before element-wise ops (could use custom kernel instead)
    cudaStreamSynchronize(stream);

    // For now, download gate and up to do element-wise multiply on CPU
    // TODO: Implement custom CUDA kernel for SILU and element-wise multiply
    std::vector<float> gate_cpu(n_ff);
    std::vector<float> up_cpu(n_ff);

    cudaMemcpy(gate_cpu.data(), cuda_ctx->gpu_gate_up_buffer.get(),
               n_ff * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(up_cpu.data(), cuda_ctx->gpu_up_buffer.get(),
               n_ff * sizeof(float), cudaMemcpyDeviceToHost);

    // Apply SILU to gate
    for (int64_t i = 0; i < n_ff; i++) {
        float sigmoid = 1.0f / (1.0f + std::exp(-gate_cpu[i]));
        gate_cpu[i] = gate_cpu[i] * sigmoid;
    }

    // Element-wise multiply
    for (int64_t i = 0; i < n_ff; i++) {
        gate_cpu[i] *= up_cpu[i];
    }

    // Copy gate_up back to GPU
    cudaMemcpyAsync(cuda_ctx->gpu_gate_up_buffer.get(), gate_cpu.data(),
                    n_ff * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Compute output = gate_up @ down_expert.T
    // gate_up: [1, n_ff], down_expert: [n_embd, n_ff]
    // down_expert.T: [n_ff, n_embd]
    // gate_up @ down_expert.T: [1, n_ff] @ [n_ff, n_embd] = [1, n_embd]
    cuda_matmul(handle,
                cuda_ctx->gpu_gate_up_buffer.get(),
                cuda_ctx->gpu_down_buffer.get(),
                cuda_ctx->gpu_output_buffer.get(),
                1, n_embd, n_ff,
                CUBLAS_OP_N, CUBLAS_OP_T);

    // Copy output back to CPU
    cudaMemcpyAsync(output, cuda_ctx->gpu_output_buffer.get(),
                    n_embd * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // Synchronize
    cudaStreamSynchronize(stream);

    // Apply expert weight
    for (int64_t i = 0; i < n_embd; i++) {
        output[i] *= expert_weight;
    }

    return true;
}

} // namespace llama
