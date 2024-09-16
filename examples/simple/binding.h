#ifndef LLAMA_ODIN_BINDING_H
#define LLAMA_ODIN_BINDING_H

#include <stdbool.h>
#include "common.h"


typedef void *gpt_params_ptr;
typedef void *llama_model_params_ptr;
typedef void *llama_model_ptr;
typedef void *llama_context_params_ptr;
typedef void *llama_context_ptr;

typedef struct Tokens {
  int32_t len;
  int32_t *tokens;
} Tokens;

#ifdef __cplusplus
extern "C" {
#endif


void lpp_print_usage(int argc, char **argv);

gpt_params_ptr lpp_make_gpt_params_ptr();

bool lpp_gpt_params_parse(int argc, char **argv, gpt_params_ptr params,
                          llama_example ex);

int32_t lpp_get_n_predict(gpt_params_ptr params);
int32_t lpp_get_n_gpu_layers(gpt_params_ptr params);
const char *lpp_get_model(gpt_params_ptr params);
const char *lpp_get_prompt(gpt_params_ptr params);
ggml_numa_strategy lpp_get_numa(gpt_params_ptr params);

llama_model_params_ptr lpp_llama_model_params_from_gpt_params(
    gpt_params_ptr params);

llama_model_ptr lpp_llama_load_model_from_file(const char *model_path,
                                               llama_model_params_ptr model_params);

llama_context_params_ptr lpp_llama_context_params_from_gpt_params(gpt_params_ptr params);

llama_context_ptr llp_llama_new_context_with_model(llama_model_ptr model, llama_context_params_ptr ctx_params);

Tokens lpp_llama_tokenize(llama_context_ptr ctx, const char* text, bool add_special, bool parse_special /*false*/);

const char *lpp_llama_token_to_piece(llama_context_ptr ctx, llama_token token,
                                     bool special /*true*/);


void lpp_llama_batch_add(llama_batch* batch, llama_token id, llama_pos pos, llama_seq_id* seq_ids, size_t seq_ids_len, bool logits);

void lpp_llama_batch_clear(llama_batch* batch);

#ifdef __cplusplus
}
#endif

#endif  // LLAMA_ODIN_BINDING_H