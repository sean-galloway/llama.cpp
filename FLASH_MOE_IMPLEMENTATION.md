# Flash-MoE Staged MoE Loading Implementation

## Overview

This implements Flash-MoE's core technique: **load only the active K experts on-demand, not all 384 at startup**.

## Problem Solved

**Native llama.cpp:**
- Loads all 384 experts × 61 layers = 23,424 expert tensors at startup
- Total: ~422GB allocation
- **Result:** OOM on DGX Spark (119GB unified memory)

**With Flash-MoE staged loading:**
- Load only attention/embeddings at startup (~10GB)
- Keep all 384 experts on disk (SSD)
- During inference, load only K=8 active experts via pread()
- Use LRU staging buffer (8 × 24MB = 192MB)
- **Result:** Fits in 119GB, GPU acceleration enabled

## Files Added

| File | Purpose |
|------|---------|
| `src/llama-staged-moe.h` | Staged loading manager and buffer |
| `src/llama-staged-moe.cpp` | Implementation with pread() loading |
| `src/llama-model-staged-moe.patch` | Integration with model loader |

## Architecture

```
Startup:
  Load: attention layers, embeddings, norms (~10GB GPU)
  Skip: all MoE expert tensors
  Store: file offsets for each expert (metadata only)

Inference:
  Router selects K=8 experts for token
  Check: Are experts in staging buffer?
  If not: pread() from SSD → CPU buffer → GPU staging
  Compute: matmul with staged experts
  LRU: Evict least-recently-used experts
```

## Key Components

### 1. staged_expert_buffer

Circular buffer holding K active experts:
```cpp
struct slot {
    int expert_id;        // Which expert (0-383)
    void* cpu_data;       // CPU buffer
    void* gpu_data;       // GPU staging buffer
    uint64_t last_used;   // For LRU eviction
};
```

### 2. staged_moe_manager

Manages all MoE layers:
```cpp
// Metadata for all experts (384 per layer × 61 layers)
std::unordered_map<std::string, std::vector<expert_metadata>> expert_meta;

// Staging buffer (K=8 experts)
staged_expert_buffer buffer;
```

### 3. Model Loader Integration

Modify `llm_load_tensors()` to:
```cpp
if (hparams.flash_moe_staged) {
    // Store file offsets, don't allocate
    for (int e = 0; e < n_expert; e++) {
        expert_metadata meta;
        meta.offset = calculate_file_offset(e);
        staged_moe_manager.store_meta(layer, type, e, meta);
    }
    // Set tensor ptr to nullptr (will be filled on-demand)
    layer.ffn_gate_exps = nullptr;
} else {
    // Original: allocate all experts
    layer.ffn_gate_exps = create_tensor(..., {n_embd, n_ff, n_expert});
}
```

### 4. Graph Integration

Modify `build_moe_ffn()` to:
```cpp
// Get selected experts from gating
int* selected_experts = get_topk_experts(...);  // K=8

// Load active experts
for (int i = 0; i < n_expert_used; i++) {
    int expert_id = selected_experts[i];
    void* expert_data = staged_moe_manager.get_expert(layer, "gate", expert_id);
    // Use in matmul
}
```

## Usage

```bash
# Build with staged loading support
cd ~/llama.cpp/build
cmake .. -DLLAMA_CUDA=ON -DLLAMA_FLASH_MOE=ON
make -j$(nproc) llama-server

# Run with Flash-MoE staged loading
export LLAMA_FLASH_MOE=1
export LLAMA_FLASH_MOE_K=8  # Number of experts to stage

./bin/llama-server \
    -m ~/models/Kimi-K2.5-Q3_K_S/Kimi-K2.5-Q3_K_S-00001-of-00010.gguf \
    -ngl 35 \
    --host 0.0.0.0 \
    --port 8000
```

## Implementation Status

### ✅ Completed
- Staged loading manager infrastructure
- pread()-based expert loading
- LRU eviction buffer
- Model loader integration plan

### ⏳ Remaining
1. Calculate exact file offsets for each expert in GGUF
2. Integrate with llama-graph.cpp build_moe_ffn()
3. Add environment variable parsing
4. Test and debug on DGX Spark
5. Optimize SSD read patterns (parallel pread)

## Performance Expectations

| Metric | Before | After |
|--------|--------|-------|
| Startup memory | 422GB (OOM) | ~10GB |
| Staging buffer | N/A | 192MB (K=8) |
| Expert loading | All at startup | On-demand via pread() |
| Expected speed | 0.8 tok/s (CPU) | 5-15 tok/s (GPU) |

## References

- [Flash-MoE](https://github.com/danveloper/flash-moe)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
