// Flash-MoE Staged MoE Loading Implementation
// Only load active experts (K=8) on-demand from disk

#include "llama-staged-moe.h"
#include "llama-model-loader.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include <ggml-backend.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

namespace llama {

// Global manager instance
static staged_moe_manager g_staged_moe_manager;
static bool g_staged_moe_enabled = false;

// staged_expert_buffer implementation
bool staged_expert_buffer::init(size_t k_experts, size_t expert_bytes, ggml_backend* cuda_backend) {
    expert_size = expert_bytes;
    backend = cuda_backend;
    slots.resize(k_experts);

    // Allocate CPU buffers for all slots
    for (auto& slot : slots) {
        slot.cpu_data = malloc(expert_size);
        if (!slot.cpu_data) {
            fprintf(stderr, "staged_expert_buffer: failed to allocate CPU buffer\n");
            return false;
        }
        slot.expert_id = -1;
    }

    // Allocate GPU buffer if backend provided
    if (backend) {
        // Note: In real implementation, use ggml backend alloc
        // For now, we'll use CPU staging and copy to GPU during compute
    }

    fprintf(stderr, "staged_expert_buffer: initialized %zu slots of %zu bytes each\n",
            k_experts, expert_bytes);
    return true;
}

void staged_expert_buffer::free() {
    for (auto& slot : slots) {
        if (slot.cpu_data) {
            ::free(slot.cpu_data);
            slot.cpu_data = nullptr;
        }
        if (slot.gpu_data) {
            // Free GPU memory if allocated
            slot.gpu_data = nullptr;
        }
    }
    slots.clear();
}

size_t staged_expert_buffer::find_lru_slot() {
    size_t lru_idx = 0;
    uint64_t min_access = access_counter;

    for (size_t i = 0; i < slots.size(); i++) {
        if (slots[i].expert_id == -1) {
            return i; // Empty slot
        }
        if (slots[i].last_used < min_access) {
            min_access = slots[i].last_used;
            lru_idx = i;
        }
    }

    return lru_idx;
}

void* staged_expert_buffer::get_expert(int expert_id, int file_fd, const expert_metadata& meta) {
    std::lock_guard<std::mutex> lock(mutex);  // Thread-safe access

    // Check if already loaded
    for (auto& slot : slots) {
        if (slot.expert_id == expert_id) {
            slot.last_used = ++access_counter;
            return slot.cpu_data;
        }
    }

    // Find slot to replace
    size_t slot_idx = find_lru_slot();
    auto& s = slots[slot_idx];

    // Load expert from disk using pread()
    ssize_t n = pread(file_fd, s.cpu_data, meta.size, (off_t)meta.offset);
    if (n != (ssize_t)meta.size) {
        fprintf(stderr, "staged_expert_buffer: pread failed for expert %d: %zd != %zu (errno=%d)\n",
                expert_id, n, meta.size, errno);
        return nullptr;
    }

    s.expert_id = expert_id;
    s.last_used = ++access_counter;

    return s.cpu_data;
}

// staged_moe_manager implementation
bool staged_moe_manager::init(const llama_model_loader& loader, const staged_moe_config& cfg) {
    config = cfg;

    // TODO: Extract expert metadata from loader
    // For each MoE layer, store file offsets for each expert

    return true;
}

bool staged_moe_manager::init(const llama_hparams& hparams, int file_fd, ggml_backend* cuda_backend) {
    config.file_fd = file_fd;

    // Calculate expert size from hparams
    // For Kimi K2.5: n_embd=7168, n_ff_exp=2048 (from config.json)
    // Expert size depends on quantization type
    // For now, use a default size
    size_t expert_elem_size = sizeof(float); // Assuming FP32 for now
    config.expert_size = (size_t)hparams.n_embd * hparams.n_ff_exp * expert_elem_size;

    // Initialize buffer with K slots
    if (!buffer.init(config.max_staged_experts, config.expert_size, cuda_backend)) {
        return false;
    }

    fprintf(stderr, "staged_moe_manager: initialized with %zu experts, size %zu bytes each\n",
            config.max_staged_experts, config.expert_size);
    return true;
}

void staged_moe_manager::register_expert(int layer, const char* type, int expert_id,
                                          uint32_t file_idx, size_t offset, size_t size) {
    char key[64];
    snprintf(key, sizeof(key), "layer_%d_%s", layer, type);

    auto& meta_vec = expert_meta[key];
    if (expert_id >= (int)meta_vec.size()) {
        meta_vec.resize(expert_id + 1);
    }

    meta_vec[expert_id].file_idx = file_idx;
    meta_vec[expert_id].offset = offset;
    meta_vec[expert_id].size = size;
    meta_vec[expert_id].is_loaded = false;
}

void* staged_moe_manager::get_expert_data(int layer, const char* type, int expert_id) {
    char key[64];
    snprintf(key, sizeof(key), "layer_%d_%s", layer, type);

    auto it = expert_meta.find(key);
    if (it == expert_meta.end() || expert_id >= (int)it->second.size()) {
        fprintf(stderr, "staged_moe_manager: expert %d not found for layer %d type %s\n",
                expert_id, layer, type);
        return nullptr;
    }

    return buffer.get_expert(expert_id, config.file_fd, it->second[expert_id]);
}

void staged_moe_manager::free() {
    buffer.free();
    expert_meta.clear();
}

// Global functions
bool staged_moe_init_from_env() {
    const char* env = std::getenv("LLAMA_FLASH_MOE");
    if (env && (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0)) {
        g_staged_moe_enabled = true;

        // Read K value from environment
        const char* k_env = std::getenv("LLAMA_FLASH_MOE_K");
        if (k_env) {
            int k = std::atoi(k_env);
            if (k > 0 && k <= 64) {
                g_staged_moe_manager.config.max_staged_experts = (size_t)k;
                fprintf(stderr, "Flash-MoE: Using K=%d experts\n", k);
            }
        }

        fprintf(stderr, "Flash-MoE: Staged loading enabled\n");
        return true;
    }
    return false;
}

bool staged_moe_is_enabled() {
    return g_staged_moe_enabled;
}

staged_moe_manager* get_staged_moe_manager() {
    return &g_staged_moe_manager;
}

void* staged_moe_get_expert_for_compute(int layer, const char* type, int expert_id) {
    if (!g_staged_moe_enabled || !g_staged_moe_manager.config.enabled) {
        return nullptr;
    }
    return g_staged_moe_manager.get_expert_data(layer, type, expert_id);
}

// Utility functions
bool is_moe_expert_tensor(const char* name) {
    return strstr(name, "ffn_gate_exps") ||
           strstr(name, "ffn_up_exps") ||
           strstr(name, "ffn_down_exps");
}

bool parse_expert_tensor_name(const char* name, int& layer, std::string& type, int& expert) {
    // Expected format: blk.{L}.ffn_{type}_exps.weight
    // e.g., "blk.5.ffn_gate_exps.weight"

    if (!is_moe_expert_tensor(name)) {
        return false;
    }

    // Parse layer number
    const char* blk = strstr(name, "blk.");
    if (!blk) return false;

    layer = std::atoi(blk + 4);

    // Parse type (gate, up, down)
    if (strstr(name, "ffn_gate_exps")) {
        type = "gate";
    } else if (strstr(name, "ffn_up_exps")) {
        type = "up";
    } else if (strstr(name, "ffn_down_exps")) {
        type = "down";
    } else {
        return false;
    }

    // Expert ID is determined by tensor dimensions, parsed elsewhere
    expert = -1; // Not parsed from name

    return true;
}

} // namespace llama
