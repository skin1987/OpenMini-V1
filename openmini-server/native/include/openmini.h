#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENMINI_VERSION_MAJOR 2
#define OPENMINI_VERSION_MINOR 0
#define OPENMINI_VERSION_PATCH 0
#define OPENMINI_VERSION_STRING "2.0.0"

typedef struct openmini_context openmini_context_t;
typedef struct openmini_model openmini_model_t;
typedef struct openmini_batch openmini_batch_t;

typedef enum {
    OPENMINI_OK = 0,
    OPENMINI_ERR_MEMORY = -1,
    OPENMINI_ERR_INVALID_PARAM = -2,
    OPENMINI_ERR_MODEL_LOAD = -3,
    OPENMINI_ERR_INFERENCE = -4,
    OPENMINI_ERR_UNSUPPORTED = -5,
    OPENMINI_ERR_CUDA = -6,
    OPENMINI_ERR_METAL = -7,
    OPENMINI_ERR_TOKENIZER = -8,
    OPENMINI_ERR_CONTEXT_FULL = -9,
} openmini_error_t;

typedef struct {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_threads;
    int gpu_device_id;
    char quant_type[16];
    bool enable_kv_cache;
    bool enable_flash_attn;
    uint32_t kv_cache_size_mb;
    bool use_bitnet_lut;
    uint32_t lut_block_size;
    float rope_scaling;
    float rope_freq_base;
    float rope_freq_scale;
} openmini_config_t;

typedef struct {
    int* tokens;
    size_t n_tokens;
    char* text;
    uint64_t duration_us;
    uint64_t prompt_tokens;
    uint64_t output_tokens;
    float tokens_per_second;
    float prompt_processing_us;
    float token_generation_us;
} openmini_result_t;

typedef struct {
    size_t n_params;
    size_t n_layers;
    size_t n_head;
    size_t n_head_kv;
    size_t n_embd;
    size_t n_vocab;
    size_t n_ff;
    char type[64];
    char quantization[16];
} openmini_model_info_t;

typedef struct {
    uint64_t total_prompt_us;
    uint64_t total_gen_us;
    uint64_t total_tokens;
    float tokens_per_second;
    float memory_used_mb;
    float gpu_memory_used_mb;
    uint32_t cache_hit_rate;
    size_t n_prompt_tokens_processed;
    size_t n_tokens_generated;
} openmini_stats_t;

openmini_error_t openmini_init(void);
void openmini_cleanup(void);
const char* openmini_version(void);
const char* const* openmini_get_backends(size_t* count);

openmini_error_t openmini_load_model(
    const char* model_path,
    const openmini_config_t* config,
    openmini_model_t** out_model
);

void openmini_free_model(openmini_model_t* model);

openmini_error_t openmini_model_info(
    const openmini_model_t* model,
    openmini_model_info_t* info
);

openmini_error_t openmini_new_context(
    const openmini_model_t* model,
    openmini_context_t** out_ctx
);

void openmini_free_context(openmini_context_t* ctx);

openmini_error_t openmini_reset_context(openmini_context_t* ctx);

openmini_error_t openmini_generate(
    openmini_context_t* ctx,
    const char* prompt,
    int max_tokens,
    float temperature,
    float top_p,
    int top_k,
    openmini_result_t* result
);

openmini_error_t openmini_decode(
    openmini_context_t* ctx,
    const int* tokens,
    size_t n_tokens,
    bool preprocess_only
);

int openmini_sample_token(
    openmini_context_t* ctx,
    float temperature,
    float top_p,
    int top_k
);

openmini_error_t openmini_get_logits(
    openmini_context_t* ctx,
    float** logits,
    size_t* n_logits
);

void openmini_free_result(openmini_result_t* result);

openmini_error_t openmini_get_stats(
    const openmini_context_t* ctx,
    openmini_stats_t* stats
);

void openmini_reset_stats(openmini_context_t* ctx);

typedef void (*openmini_log_callback_t)(int level, const char* text, void* user_data);
openmini_error_t openmini_set_log_callback(
    openmini_log_callback_t callback,
    void* user_data
);

#ifdef __cplusplus
}
#endif
