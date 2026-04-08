# Flash-MoE Integration Guide

## Overview

This directory contains all the patches and files needed to integrate Flash-MoE staged loading into llama.cpp.

## Files

### Core Implementation
- `llama-staged-moe-complete.h` - Header with staging manager and buffer
- `llama-staged-moe-complete.cpp` - Implementation with pread() loading

### Integration Patches
- `llama-model-flash-moe-integration.patch` - Patches model loader to defer expert allocation
- `llama-graph-flash-moe-integration.patch` - Patches graph to load experts on-demand
- `CMakeLists-flash-moe.patch` - Adds new source files to build

### Build Script
- `build-flash-moe.sh` - Applies patches and builds llama.cpp

## Quick Start

```bash
cd ~/llama.cpp

# Build with Flash-MoE support
./build-flash-moe.sh

# Run with staged loading
export LLAMA_FLASH_MOE=1
export LLAMA_FLASH_MOE_K=8
./build/bin/llama-server -m model.gguf -ngl 35
```

## What Gets Modified

### 1. Model Loading (`llama-model.cpp`)
When `LLAMA_FLASH_MOE=1` is set:
- Expert tensors are NOT allocated at startup
- File offsets are stored for each expert
- Tensor pointers are set to nullptr

### 2. Graph Execution (`llama-graph.cpp`)
Before each MoE layer computation:
- Router selects K active experts
- `staged_moe_prepare_experts()` loads them via pread()
- Experts are copied to GPU staging buffer
- Computation proceeds with staged experts

### 3. Memory Layout
```
Without Flash-MoE:
  GPU: 422GB (all experts + attention) → OOM

With Flash-MoE:
  GPU: ~10GB (attention only)
  Disk: 413GB (all experts on SSD)
  Staging: 192MB (K=8 experts loaded on-demand)
```

## Testing

```bash
# Test with small model first
export LLAMA_FLASH_MOE=1
./build/bin/llama-server -m small-model.gguf -ngl 10

# Check logs for Flash-MoE messages
tail -f /tmp/llama.log | grep "Flash-MoE"
```

## Troubleshooting

### "Flash-MoE: pread failed"
- Check file permissions on GGUF
- Verify SSD has enough free space
- Check `dmesg` for disk errors

### Still getting OOM
- Reduce `-ngl` value
- Reduce context size with `-c 2048`
- Check `nvidia-smi` for other GPU processes

### Slow performance
- Check SSD speed: `fio --name=test --rw=read --bs=1M --size=1G`
- Should be >3GB/s sequential read
- Warm up page cache by reading model once

## Architecture

```
Startup:
  llama-model.cpp:
    if LLAMA_FLASH_MOE=1:
      Skip expert tensor allocation
      Store file offsets in staged_moe_manager

Inference (per token):
  llama-graph.cpp build_moe_ffn():
    1. Router computes expert selection → K expert IDs
    2. staged_moe_prepare_experts():
       For each selected expert:
         - Check if in staging buffer
         - If not: pread() from SSD → CPU buffer
         - Copy to GPU staging buffer
    3. Execute matmul with staged experts
    4. LRU eviction for next token
```

## Implementation Notes

### Expert Loading
- Uses `posix_memalign()` for O_DIRECT compatible buffers
- `pread()` for position-independent, thread-safe reads
- LRU eviction when staging buffer is full

### GPU Integration
- Experts are loaded to CPU staging buffer first
- Then copied to GPU via `cudaMemcpy()` or ggml backend
- Small CUDA allocation (~192MB) avoids MMU faults

### File Descriptor Management
- GGUF file is kept open during entire session
- Each `pread()` uses the saved file descriptor
- File is closed when model is unloaded

## Future Improvements

1. **Async Loading**: Load next predicted experts while computing current
2. **Batched pread()**: Load multiple experts in parallel
3. **Compression**: Use INT4/INT2 quantization for experts
4. **Prefetching**: Predict which experts will be needed next

## References

- [Flash-MoE](https://github.com/danveloper/flash-moe)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
