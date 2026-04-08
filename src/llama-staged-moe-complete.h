// Flash-MoE Staged MoE Loading - Complete Header
// Loads only active K=8 experts on-demand from disk

#ifndef LLAMA_STAGED_MOE_H
#define LLAMA_STAGED_MOE_H

#include "ggml.h"
#include "llama.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstring>

// Forward declarations
struct llama_hparams;
struct llama_model_loader;

namespace llama {

// Configuration for staged MoE loading
struct staged_moe_config {
    bool enabled = false;           // Enable staged loading
    size_t max_staged_experts = 8;  // Number of experts to keep staged (K value)
    size_t expert_size = 0;         // Size of one expert in bytes
    int file_fd = -1;               // File descriptor for pread()
};

// Expert metadata for lazy loading
struct expert_metadata {
    uint32_t file_idx = 0;      // Which GGUF file
    size_t offset = 0;          // Byte offset in file
    size_t size = 0;            // Size in bytes
    bool is_loaded = false;     // Currently in memory?
};

// Staged expert buffer - circular buffer for active experts
struct staged_expert_buffer {
    struct slot {
        int expert_id = -1;         // Which expert (0-383)
        void* cpu_data = nullptr;   // CPU buffer (aligned for O_DIRECT)
        uint64_t last_used = 0;     // For LRU eviction
    };

    std::vector<slot> slots;
    size_t expert_size = 0;
    size_t max_experts = 8;
    uint64_t access_counter = 0;
    ggml_backend* backend = nullptr;

    // Initialize with K slots
    bool init(size_t k_experts, size_t expert_bytes, ggml_backend* cuda_backend);
    void free();

    // Get expert - loads if not present using pread()
    void* get_expert(int expert_id, int file_fd, const expert_metadata& meta);

    // Find LRU slot for replacement
    size_t find_lru_slot();
};

// Global staged MoE manager
struct staged_moe_manager {
    staged_moe_config config;
    staged_expert_buffer buffer;

    // Metadata for all experts per layer
    // Key: "layer_{L}_{type}" where type is gate/up/down
    std::unordered_map<std::string, std::vector<expert_metadata>> expert_meta;

    // Initialize from hparams and open file
    bool init(const llama_hparams& hparams, int file_fd, ggml_backend* cuda_backend);

    // Register expert metadata (called during model loading)
    void register_expert(int layer, const char* type, int expert_id,
                         uint32_t file_idx, size_t offset, size_t size);

    // Get expert data - loads via pread() if not cached
    void* get_expert_data(int layer, const char* type, int expert_id);

    // Shutdown
    void free();
};

// Initialize from environment variables
// Call this early in llama_init
bool staged_moe_init_from_env();

// Check if staged loading is enabled
bool staged_moe_is_enabled();

// Get the global staged MoE manager
staged_moe_manager* get_staged_moe_manager();

// Hook for build_moe_ffn - get expert pointer for computation
void* staged_moe_get_expert_for_compute(int layer, const char* type, int expert_id);

// Check if tensor name is an MoE expert tensor
bool is_moe_expert_tensor(const char* name);

} // namespace llama

#endif // LLAMA_STAGED_MOE_H
