# Flash-MoE pread() Patch for llama.cpp

This patch implements Flash-MoE's pread() + GPU staging buffer technique for llama.cpp to enable GPU acceleration on unified memory systems (like DGX Spark GB10) without MMU faults.

## Problem Solved

**Native llama.cpp:**
- Uses `mmap()` to load model weights
- GPU accesses mmap'd memory directly
- On unified memory systems (DGX Spark, Apple Silicon), OS can evict pages
- **Result:** MMU faults (Xid 31) when GPU tries to access evicted pages
- **Workaround:** CPU-only mode (~0.8 tok/s)

**This Patch:**
- Uses `pread()` for explicit, on-demand expert loading
- Small GPU staging buffer (~100-200MB) for active experts only
- Experts live in regular CPU memory (`malloc`), not GPU-mapped memory
- **Result:** No MMU faults, GPU acceleration enabled
- **Expected:** 5-15 tok/s (10-20x improvement)

## How It Works

```
Native llama.cpp:
  mmap(413GB model) → GPU pointer to mmap'd memory
  → OS evicts pages → MMU FAULT → Crash

With pread() patch:
  malloc(100MB staging) → pread(K=6 active experts × 24MB)
  → cudaMemcpy() to GPU staging → GPU compute → Discard
```

## Files Added

| File | Description |
|------|-------------|
| `llama-pread-staging.h` | Header for staging buffer and pread loader |
| `llama-pread-staging.cpp` | Implementation of staging buffer |
| `llama-pread.patch` | Patch for llama.cpp integration |

## Usage

### 1. Apply Patch

```bash
cd ~/llama.cpp
patch -p1 < /path/to/llama-pread.patch
cp /path/to/llama-pread-staging.h src/
cp /path/to/llama-pread-staging.cpp src/
```

### 2. Rebuild

```bash
cd ~/llama.cpp/build
make -j$(nproc)
```

### 3. Run with Environment Variables

```bash
# Enable pread-based loading
export LLAMA_USE_PREAD=1

# Enable GPU staging buffer
export LLAMA_USE_GPU_STAGING=1

# Optional: Customize staging buffer size (default: 144MB for K=6)
export LLAMA_STAGING_BUFFER_SIZE=200  # MB

# Run llama.cpp server with GPU acceleration
./bin/llama-server \
    -m ~/models/Kimi-K2.5-GGUF/Q3_K_S/Kimi-K2.5-Q3_K_S-00001-of-00010.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    -ngl 999  # Use all GPU layers (now safe!)
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LLAMA_USE_PREAD` | 0 | Enable pread() instead of mmap |
| `LLAMA_USE_GPU_STAGING` | 0 | Enable GPU staging buffer |
| `LLAMA_STAGING_BUFFER_SIZE` | 144 | Staging buffer size in MB |

## Expected Performance

| Configuration | Speed | Notes |
|---------------|-------|-------|
| Native CPU-only | ~0.8 tok/s | Current working config |
| Native GPU (broken) | Crashes | MMU faults |
| With pread() patch | 5-15 tok/s | GPU acceleration working |

## Technical Details

### Expert Loading Flow

1. **Gating:** Router selects K=6 experts for each token
2. **Cache Check:** Check if experts are already in staging buffer
3. **pread():** Load missing experts from GGUF file
4. **cudaMemcpy():** Copy to GPU staging buffer
5. **Compute:** Run matmul with staged experts
6. **LRU Eviction:** Replace least-recently-used experts

### Memory Layout

```
CPU Memory (malloc):
  - File cache: OS page cache for GGUF file
  - Bounce buffer: Temporary for pread()

GPU Memory (cudaMalloc):
  - Staging buffer: 100-200MB for K active experts
  - Attention layers: Full attention compute
  - KV cache: Attention key/value cache
```

## Limitations

1. **Currently supports MoE models only** (Kimi K2.5, Mixtral, etc.)
2. **Requires GGUF format** (not safetensors)
3. **Staging buffer size limits concurrent experts** (default K=6 fits)
4. **File I/O overhead** vs mmap (offset by 10-20x GPU speedup)

## Troubleshooting

### Still getting MMU faults
- Verify `LLAMA_USE_PREAD=1` and `LLAMA_USE_GPU_STAGING=1`
- Check that model is GGUF format
- Ensure sufficient staging buffer size

### Slow performance
- Check SSD speed: should be >3GB/s sequential read
- Verify OS page cache is warmed up
- Try increasing staging buffer size

### Out of memory
- Reduce staging buffer size
- Reduce batch size
- Use smaller quantization (Q4 instead of Q8)

## References

- [Flash-MoE Paper](https://github.com/danveloper/flash-moe)
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [DGX Spark Documentation](https://www.nvidia.com/en-us/data-center/dgx-spark/)
