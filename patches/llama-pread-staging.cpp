// Flash-MoE inspired pread() + GPU staging buffer implementation
// Avoids MMU faults on unified memory systems by using explicit loading

#include "llama-pread-staging.h"
#include <ggml.h>
#include <ggml-backend.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>
#include <cstdio>

namespace llama {

// expert_staging_buffer implementation
bool expert_staging_buffer::init(size_t expert_bytes, size_t k_active, ggml_backend* cuda_backend) {
    expert_size = expert_bytes;
    max_experts = k_active;
    buffer_size = expert_size * max_experts;
    backend = cuda_backend;

    if (!backend) {
        fprintf(stderr, "staging_buffer: no CUDA backend provided\n");
        return false;
    }

    // Allocate CUDA staging buffer
    // This is the key: small, explicit allocation that won't cause MMU faults
    cuda_buffer = ggml_backend_buffer_get_base(ggml_backend_cpu_buffer_from_ptr(nullptr, buffer_size));

    // Actually, we need proper CUDA allocation
    // Use ggml backend for proper memory management
    ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(backend, buffer_size);
    if (!buf) {
        fprintf(stderr, "staging_buffer: failed to allocate %zu bytes on GPU\n", buffer_size);
        return false;
    }
    cuda_buffer = ggml_backend_buffer_get_base(buf);

    // Allocate CPU bounce buffer for pread
    cpu_buffer = malloc(buffer_size);
    if (!cpu_buffer) {
        fprintf(stderr, "staging_buffer: failed to allocate CPU bounce buffer\n");
        return false;
    }

    fprintf(stderr, "staging_buffer: initialized with %zu bytes (%zu experts × %zu bytes)\n",
            buffer_size, max_experts, expert_size);
    return true;
}

void expert_staging_buffer::free() {
    if (cuda_buffer) {
        // Buffer is managed by ggml backend
        cuda_buffer = nullptr;
    }
    if (cpu_buffer) {
        ::free(cpu_buffer);
        cpu_buffer = nullptr;
    }
    buffer_size = 0;
}

void* expert_staging_buffer::stage_expert(int expert_id, const void* file_data, size_t offset) {
    if (!cuda_buffer || !cpu_buffer) {
        return nullptr;
    }

    // Calculate slot in staging buffer (round-robin)
    size_t slot = expert_id % max_experts;
    void* cpu_slot = (uint8_t*)cpu_buffer + slot * expert_size;
    void* gpu_slot = (uint8_t*)cuda_buffer + slot * expert_size;

    // Copy from file to CPU bounce buffer
    // In real implementation, this would use pread() from file descriptor
    memcpy(cpu_slot, (const uint8_t*)file_data + offset, expert_size);

    // Async copy to GPU staging buffer
    // This is non-blocking; GPU compute can overlap with next expert load
    ggml_backend_tensor_set_async(backend,
        ggml_new_tensor_1d(nullptr, GGML_TYPE_F32, expert_size / sizeof(float)),
        cpu_slot, 0, expert_size);

    return gpu_slot;
}

bool expert_staging_buffer::is_staged(int expert_id) const {
    // Simple check - in production, track which slots have which experts
    return false;
}

void* expert_staging_buffer::get_gpu_ptr(int expert_id) const {
    if (!cuda_buffer) return nullptr;
    size_t slot = expert_id % max_experts;
    return (uint8_t*)cuda_buffer + slot * expert_size;
}

// pread_model_loader implementation
bool pread_model_loader::init(const char* filepath) {
    file_fd = open(filepath, O_RDONLY | O_DIRECT);
    if (file_fd < 0) {
        fprintf(stderr, "pread_loader: failed to open %s\n", filepath);
        return false;
    }

    struct stat st;
    if (fstat(file_fd, &st) < 0) {
        fprintf(stderr, "pread_loader: failed to stat file\n");
        close(file_fd);
        file_fd = -1;
        return false;
    }
    file_size = st.st_size;

    // Allocate CPU buffer for reads (aligned for O_DIRECT)
    size_t buf_size = 64 * 1024 * 1024; // 64MB chunks
    if (posix_memalign((void**)&cpu_buffer.data(), 4096, buf_size) != 0) {
        fprintf(stderr, "pread_loader: failed to allocate aligned buffer\n");
        close();
        return false;
    }

    fprintf(stderr, "pread_loader: opened %s (%zu bytes)\n", filepath, file_size);
    return true;
}

void pread_model_loader::close() {
    if (file_fd >= 0) {
        ::close(file_fd);
        file_fd = -1;
    }
    if (!cpu_buffer.empty()) {
        free(cpu_buffer.data());
        cpu_buffer.clear();
    }
    file_size = 0;
}

const void* pread_model_loader::read_at(size_t offset, size_t size) {
    if (file_fd < 0 || offset + size > file_size) {
        return nullptr;
    }

    // Resize buffer if needed
    if (size > cpu_buffer.size()) {
        cpu_buffer.resize(size);
    }

    // Use pread() for thread-safe, position-independent reads
    ssize_t n = pread(file_fd, cpu_buffer.data(), size, offset);
    if (n != (ssize_t)size) {
        fprintf(stderr, "pread_loader: pread failed at offset %zu\n", offset);
        return nullptr;
    }

    return cpu_buffer.data();
}

bool pread_model_loader::read_expert(size_t offset, size_t size, void* dst) {
    if (file_fd < 0 || offset + size > file_size) {
        return false;
    }

    ssize_t n = pread(file_fd, dst, size, offset);
    return n == (ssize_t)size;
}

// moe_expert_cache implementation
bool moe_expert_cache::init(size_t expert_bytes, size_t k_active, ggml_backend* cuda_backend) {
    if (!staging.init(expert_bytes, k_active, cuda_backend)) {
        return false;
    }
    cache.resize(k_active);
    fprintf(stderr, "moe_cache: initialized for %zu concurrent experts\n", k_active);
    return true;
}

void* moe_expert_cache::get_expert(int expert_id, pread_model_loader& loader, size_t file_offset) {
    // Check if already cached
    for (auto& ce : cache) {
        if (ce.expert_id == expert_id) {
            ce.last_used = ++access_counter;
            return ce.gpu_data;
        }
    }

    // Find LRU slot
    size_t lru_idx = 0;
    uint64_t lru_count = access_counter;
    for (size_t i = 0; i < cache.size(); i++) {
        if (cache[i].last_used < lru_count) {
            lru_count = cache[i].last_used;
            lru_idx = i;
        }
    }

    // Load expert using pread
    size_t expert_size = staging.expert_size;
    std::vector<uint8_t> temp_buf(expert_size);
    if (!loader.read_expert(file_offset, expert_size, temp_buf.data())) {
        fprintf(stderr, "moe_cache: failed to load expert %d\n", expert_id);
        return nullptr;
    }

    // Stage to GPU
    void* gpu_ptr = staging.stage_expert(expert_id, temp_buf.data(), 0);

    // Update cache entry
    cache[lru_idx].expert_id = expert_id;
    cache[lru_idx].gpu_data = gpu_ptr;
    cache[lru_idx].last_used = ++access_counter;

    return gpu_ptr;
}

void moe_expert_cache::clear() {
    for (auto& ce : cache) {
        ce.expert_id = -1;
        ce.gpu_data = nullptr;
        ce.last_used = 0;
    }
    access_counter = 0;
}

} // namespace llama
