// Flash-MoE Simple Integration
// Minimal implementation to verify environment variable support

#include "flash-moe-simple.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

bool flash_moe_enabled(void) {
    const char* env = getenv("LLAMA_FLASH_MOE");
    return env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0);
}

int flash_moe_get_k(void) {
    const char* env = getenv("LLAMA_FLASH_MOE_K");
    if (env) {
        return atoi(env);
    }
    return 8; // Default K=8
}

void flash_moe_log_status(void) {
    if (flash_moe_enabled()) {
        printf("Flash-MoE: ENABLED (K=%d)\n", flash_moe_get_k());
    } else {
        printf("Flash-MoE: disabled\n");
    }
}
