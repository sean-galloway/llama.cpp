# --no-mmap Test Results on DGX Spark

## Summary

Testing `--no-mmap` flag on DGX Spark (ARM64/Blackwell) with Kimi K2.5 (413GB GGUF).

## Test Results

### Test 1: Minimal GPU Layers
```bash
llama-server \
  -m ~/models/Kimi-K2.5-Q3_K_S/Kimi-K2.5-Q3_K_S-00001-of-00010.gguf \
  --no-mmap \
  -ngl 1 \
  -c 2048
```

**Result:** ❌ OOM
```
ggml_aligned_malloc: insufficient memory (attempted to allocate 422618.94 MB)
```

### Test 2: CPU MoE + GPU Attention
```bash
llama-server \
  -m ~/models/Kimi-K2.5-Q3_K_S/Kimi-K2.5-Q3_K_S-00001-of-00010.gguf \
  --no-mmap \
  --cpu-moe \
  -ngl 10 \
  -c 2048
```

**Result:** ❌ OOM
```
ggml_aligned_malloc: insufficient memory (attempted to allocate 422618.94 MB)
```

## Key Findings

1. **--no-mmap works** - No MMU faults, model starts loading
2. **Allocation at startup** - llama.cpp allocates ALL expert tensors at load time
3. **Flags don't help** - `-ngl` and `--cpu-moe` only affect placement, not timing
4. **422GB allocation** - Total model size exceeds DGX Spark's 119GB unified memory

## Root Cause

llama.cpp's tensor loading flow:
1. Parse GGUF metadata
2. Allocate ALL tensors (422GB) in `load_tensors()`
3. Copy data from file to allocated buffers

The `--no-mmap` flag changes step 3 (copy vs mmap), but step 2 still allocates everything.

## Solution: Staged Loading

Need to modify llama.cpp to:
1. Parse GGUF metadata (don't change)
2. Allocate NON-expert tensors only (attention, embeddings, etc. ~10GB)
3. Skip expert tensor allocation at startup
4. Load experts on-demand during `build_moe_ffn()` using pread()
5. Use GPU staging buffer for active experts (K=6 × 24MB = 144MB)

## Implementation Plan

See `PREAD_INTEGRATION.md` for technical details.

### Files to Modify

1. `src/llama-model-loader.cpp`
   - Add flag to defer expert tensor allocation
   - Store file offsets for lazy loading

2. `src/llama-graph.cpp`
   - Hook into `build_moe_ffn()`
   - Load active experts via pread()
   - Manage GPU staging buffer

3. `src/llama-pread-staging.cpp` (new)
   - GPU staging buffer management
   - LRU cache for experts
   - pread() loading

## Next Steps

1. Implement staged loading in model loader
2. Test with `-ngl 10` (attention on GPU, experts staged)
3. Measure throughput improvement vs CPU-only
4. Sync changes to dgx2 and test RPC mode
