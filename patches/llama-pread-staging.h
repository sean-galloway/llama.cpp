// Flash-MoE inspired pread() + GPU staging buffer for llama.cpp
// This header adds support for on-demand expert loading without mmap

#ifndef LLAMA_PREAD_STAGING_H
#define LLAMA_PREAD_STAGING_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>

// Forward declarations
struct ggml_tensor;
struct ggml_backend;

namespace llama {

// GPU staging buffer for active experts
// This avoids MMU faults by using a small CUDA buffer instead of mmap
struct expert_staging_buffer {
    // Configuration
    size_t expert_size = 0;      // Size of one expert in bytes
    size_t max_experts = 8;      // Maximum concurrent experts (K value)
    size_t buffer_size = 0;      // Total staging buffer size

    // CUDA memory
    void* cuda_buffer = nullptr; // GPU staging buffer
    void* cpu_buffer = nullptr;  // CPU bounce buffer for pread

    // Backend
    ggml_backend* backend = nullptr;

    // Initialize staging buffer
    bool init(size_t expert_bytes, size_t k_active, ggml_backend* cuda_backend);

    // Cleanup
    void free();

    // Load expert into staging buffer
    // Returns GPU pointer to staged expert
    void* stage_expert(int expert_id, const void* file_data, size_t offset);

    // Check if expert is already staged
    bool is_staged(int expert_id) const;

    // Get GPU pointer for staged expert
    void* get_gpu_ptr(int expert_id) const;
};

// Pread-based model loader
// Alternative to mmap for unified memory systems
struct pread_model_loader {
    int file_fd = -1;                    // File descriptor for pread
    size_t file_size = 0;                // Total file size
    std::vector<uint8_t> cpu_buffer;     // Buffer for pread operations

    // Initialize with file path
    bool init(const char* filepath);

    // Cleanup
    void close();

    // Read data at offset using pread()
    // Returns pointer to loaded data (in cpu_buffer)
    const void* read_at(size_t offset, size_t size);

    // Read expert data directly into user buffer
    bool read_expert(size_t offset, size_t size, void* dst);
};

// MoE expert cache with pread loading
struct moe_expert_cache {
    struct cached_expert {
        int expert_id = -1;
        void* gpu_data = nullptr;
        uint64_t last_used = 0;
    };

    expert_staging_buffer staging;
    std::vector<cached_expert> cache;
    uint64_t access_counter = 0;

    // Initialize cache for given expert count and size
    bool init(size_t expert_bytes, size_t k_active, ggml_backend* cuda_backend);

    // Get expert - loads if not cached
    void* get_expert(int expert_id, pread_model_loader& loader, size_t file_offset);

    // Clear cache
    void clear();
};

} // namespace llama

#endif // LLAMA_PREAD_STAGING_H
