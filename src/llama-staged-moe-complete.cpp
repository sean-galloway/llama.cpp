// Flash-MoE Staged MoE Loading - Complete Implementation
// Loads only active K=8 experts on-demand from disk

#include "llama-staged-moe.h"
#include "llama-model.h"
#include <ggml-backend.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>

namespace llama {

// Global staged MoE manager instance
static staged_moe_manager g_staged_moe;

// Initialize from environment variables
bool staged_moe_init_from_env() {
    const char* enabled = getenv("LLAMA_FLASH_MOE");
    if (!enabled || (strcmp(enabled, "1") != 0 && strcmp(enabled, "true") != 0)) {
        return false;
    }

    fprintf(stderr, "Flash-MoE: Staged loading enabled\n");

    g_staged_moe.config.enabled = true;

    const char* k_val = getenv("LLAMA_FLASH_MOE_K");
    if (k_val) {
        g_staged_moe.config.max_staged_experts = atoi(k_val);
        fprintf(stderr, "Flash-MoE: K=%zu experts will be staged\n",
                g_staged_moe.config.max_staged_experts);
    } else {
        g_staged_moe.config.max_staged_experts = 8; // Default K=8
    }

    return true;
}

// Check if staged loading is enabled
bool staged_moe_is_enabled() {
    return g_staged_moe.config.enabled;
}

// Get the global staged MoE manager
staged_moe_manager* get_staged_moe_manager() {
    return &g_staged_moe;
}

// staged_expert_buffer implementation
bool staged_expert_buffer::init(size_t k_experts, size_t expert_bytes, ggml_backend* cuda_backend) {
    expert_size = expert_bytes;
    max_experts = k_experts;
    backend = cuda_backend;
    slots.resize(k_experts);

    // Allocate aligned CPU buffers for efficient pread()
    for (auto& slot : slots) {
        if (posix_memalign(&slot.cpu_data, 4096, expert_size) != 0) {
            fprintf(stderr, "Flash-MoE: failed to allocate aligned buffer\n");
            return false;
        }
        slot.expert_id = -1;
        slot.last_used = 0;
    }

    fprintf(stderr, "Flash-MoE: initialized %zu slots of %zu bytes each = %zu MB total\n",
            k_experts, expert_bytes, (k_experts * expert_bytes) / (1024*1024));
    return true;
}

void staged_expert_buffer::free() {
    for (auto& slot : slots) {
        if (slot.cpu_data) {
            ::free(slot.cpu_data);
            slot.cpu_data = nullptr;
        }
    }
    slots.clear();
}

size_t staged_expert_buffer::find_lru_slot() {
    size_t lru_idx = 0;
    uint64_t min_access = access_counter;

    for (size_t i = 0; i < slots.size(); i++) {
        if (slots[i].expert_id == -1) {
            return i; // Empty slot available
        }
        if (slots[i].last_used < min_access) {
            min_access = slots[i].last_used;
            lru_idx = i;
        }
    }

    return lru_idx;
}

void* staged_expert_buffer::get_expert(int expert_id, int file_fd, const expert_metadata& meta) {
    // Check if already in buffer
    for (auto& slot : slots) {
        if (slot.expert_id == expert_id) {
            slot.last_used = ++access_counter;
            return slot.cpu_data;
        }
    }

    // Find slot to replace
    size_t slot_idx = find_lru_slot();
    slot& s = slots[slot_idx];

    // Load expert using pread() - this is the Flash-MoE technique!
    ssize_t n = pread(file_fd, s.cpu_data, meta.size, meta.offset);
    if (n != (ssize_t)meta.size) {
        fprintf(stderr, "Flash-MoE: pread failed for expert %d (expected %zu, got %zd)\n",
                expert_id, meta.size, n);
        return nullptr;
    }

    s.expert_id = expert_id;
    s.last_used = ++access_counter;

    // Track stats
    static size_t total_loads = 0;
    total_loads++;
    if (total_loads % 100 == 0) {
        fprintf(stderr, "Flash-MoE: loaded %zu experts from disk\n", total_loads);
    }

    return s.cpu_data;
}

// staged_moe_manager implementation
bool staged_moe_manager::init(const llama_hparams& hparams, int file_fd, ggml_backend* cuda_backend) {
    if (!config.enabled) {
        return false;
    }

    config.file_fd = file_fd;

    // Calculate expert size: each expert is [n_embd, n_ff] float matrix
    // For Kimi K2.5: n_embd=7168, n_ff_exp=2048
    size_t n_embd = hparams.n_embd;
    size_t n_ff = hparams.n_ff_exp > 0 ? hparams.n_ff_exp : hparams.n_ff;
    config.expert_size = n_embd * n_ff * sizeof(float);

    // For gate and up experts, size is the same
    // For down experts: [n_ff, n_embd]

    fprintf(stderr, "Flash-MoE: expert size = %zu bytes (%zu x %zu)\n",
            config.expert_size, n_embd, n_ff);
    fprintf(stderr, "Flash-MoE: total experts = %d per layer x %d layers = %d\n",
            hparams.n_expert, hparams.n_layer, hparams.n_expert * hparams.n_layer);

    // Initialize staging buffer
    if (!buffer.init(config.max_staged_experts, config.expert_size, cuda_backend)) {
        return false;
    }

    return true;
}

void staged_moe_manager::register_expert(int layer, const char* type, int expert_id,
                                          uint32_t file_idx, size_t offset, size_t size) {
    char key[64];
    snprintf(key, sizeof(key), "layer_%d_%s", layer, type);

    expert_metadata meta;
    meta.file_idx = file_idx;
    meta.offset = offset;
    meta.size = size;
    meta.is_loaded = false;

    // Ensure vector is large enough
    if (expert_meta[key].size() <= (size_t)expert_id) {
        expert_meta[key].resize(expert_id + 1);
    }
    expert_meta[key][expert_id] = meta;
}

void* staged_moe_manager::get_expert_data(int layer, const char* type, int expert_id) {
    if (!config.enabled || config.file_fd < 0) {
        return nullptr;
    }

    char key[64];
    snprintf(key, sizeof(key), "layer_%d_%s", layer, type);

    auto it = expert_meta.find(key);
    if (it == expert_meta.end() || expert_id >= (int)it->second.size()) {
        fprintf(stderr, "Flash-MoE: expert metadata not found for %s[%d]\n", key, expert_id);
        return nullptr;
    }

    return buffer.get_expert(expert_id, config.file_fd, it->second[expert_id]);
}

void staged_moe_manager::free() {
    buffer.free();
    expert_meta.clear();
    if (config.file_fd >= 0) {
        ::close(config.file_fd);
        config.file_fd = -1;
    }
}

// Hook for build_moe_ffn to load staged experts
// This is called during graph construction to get expert pointers
void* staged_moe_get_expert_for_compute(int layer, const char* type, int expert_id) {
    if (!g_staged_moe.config.enabled) {
        return nullptr;
    }
    return g_staged_moe.get_expert_data(layer, type, expert_id);
}

} // namespace llama
