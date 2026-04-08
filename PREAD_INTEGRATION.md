# Flash-MoE pread() Integration Plan

## Current Status

✅ Fork created: `sean-galloway/llama.cpp`
✅ Branch: `flash-moe-pread`
✅ Staging buffer files added
⏳ Integration with model loader needed

## Integration Approach

Since the llama.cpp codebase structure differs from the patch, here's the integration plan:

### 1. Modify `src/llama-model-loader.cpp`

Add an alternative loading path that uses pread() instead of mmap for MoE expert tensors:

```cpp
// Around line 1327 - in the tensor loading logic
// When use_mmap is false and tensor is an MoE expert:
// - Don't allocate full tensor at startup
// - Store file offset for later loading
// - Use pread() on-demand during inference
```

### 2. Modify `src/llama-graph.cpp`

Hook into `build_moe_ffn()` (line 1226) to:
- Detect which experts are active (K=6)
- Load them via pread() into staging buffer
- Use staged experts for computation

### 3. Environment Variables

Add support for:
- `LLAMA_USE_PREAD=1` - Use pread() for MoE experts
- `LLAMA_STAGING_SIZE_MB=200` - Staging buffer size

## Testing on DGX Sparks

### On dgx1:
```bash
cd ~/llama.cpp
git fetch origin
git checkout flash-moe-pread

# Build with patches applied
cd build
cmake .. -DLLAMA_CUDA=ON
make -j$(nproc) llama-server

# Test with no-mmap and GPU layers
export LLAMA_USE_PREAD=1
./bin/llama-server \
  -m ~/models/Kimi-K2.5-Q3_K_S/Kimi-K2.5-Q3_K_S-*.gguf \
  --no-mmap \
  -ngl 10 \
  --host 0.0.0.0 \
  --port 8888
```

## Simpler Alternative

Instead of full integration, we could:

1. Use `--no-mmap` flag (already works, no MMU faults)
2. Reduce memory footprint by:
   - Loading only attention layers on GPU (`-ngl 10`)
   - Keeping experts on CPU
   - Using smaller context size (`-c 2048`)

3. If that fails due to memory, try REAP-pruned model (289GB)

## Next Steps

1. Test current `--no-mmap` behavior with various `-ngl` values
2. If OOM, implement staged loading in model loader
3. Push changes to `flash-moe-pread` branch
4. Test on dgx1/dgx2 with RPC
