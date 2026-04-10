// Flash-MoE Simple Integration Header
// Minimal integration for staged expert loading

#ifndef FLASH_MOE_SIMPLE_H
#define FLASH_MOE_SIMPLE_H

#include <cstddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Check if Flash-MoE is enabled via environment variable
bool flash_moe_enabled(void);

// Get K value (number of experts to stage)
int flash_moe_get_k(void);

// Log Flash-MoE status
void flash_moe_log_status(void);

#ifdef __cplusplus
}
#endif

#endif // FLASH_MOE_SIMPLE_H
