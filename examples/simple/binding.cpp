#include "binding.h"
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

void lpp_print_usage(int argc, char **argv){
    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s -m model.gguf -p \"Hello my name is\" -n 32\n", argv[0]);
    LOG_TEE("\n");
}

gpt_params_ptr lpp_make_gpt_params_ptr() {
  return new gpt_params{};
}

bool lpp_gpt_params_parse(int argc, char **argv, gpt_params_ptr params,
                          llama_example ex) {
  gpt_params *p = static_cast<gpt_params *>(params);
  return gpt_params_parse(argc, argv, *p, ex, lpp_print_usage);
}

int32_t lpp_get_n_predict(gpt_params_ptr params) {
    gpt_params *p = static_cast<gpt_params *>(params);
    return p->n_predict;
}
int32_t lpp_get_n_gpu_layers(gpt_params_ptr params) {
    gpt_params *p = static_cast<gpt_params *>(params);
    return p->n_gpu_layers;
}
const char *lpp_get_model(gpt_params_ptr params) {
    gpt_params *p = static_cast<gpt_params *>(params);
    return p->model.c_str();
}

const char *lpp_get_prompt(gpt_params_ptr params) {
    gpt_params *p = static_cast<gpt_params *>(params);
    return  p->prompt.c_str();
}

ggml_numa_strategy lpp_get_numa(gpt_params_ptr params) {
    gpt_params *p = static_cast<gpt_params *>(params);
    return p->numa;
}

llama_model_params_ptr lpp_llama_model_params_from_gpt_params(gpt_params_ptr params) {
    gpt_params *p = static_cast<gpt_params *>(params);
    return new llama_model_params{ llama_model_params_from_gpt_params(*p) };
}

llama_model_ptr lpp_llama_load_model_from_file(const char *model_path,
                                               llama_model_params_ptr model_params) {
  llama_model_params *mp = static_cast<llama_model_params *>(model_params);
  return llama_load_model_from_file(model_path, *mp);
}

// llama_context_params ctx_params = llama_context_params_from_gpt_params(params)
llama_context_params_ptr lpp_llama_context_params_from_gpt_params(gpt_params_ptr params) {
    gpt_params *p = static_cast<gpt_params *>(params);
    return new llama_context_params{llama_context_params_from_gpt_params(*p)};
}

llama_context_ptr llp_llama_new_context_with_model(llama_model_ptr model, llama_context_params_ptr ctx_params) {
  llama_model *lm = static_cast<llama_model *>(model);
  llama_context_params* cp = static_cast<llama_context_params *>(ctx_params);
  llama_context * ctx = llama_new_context_with_model(lm, *cp);
  return ctx;
}

Tokens lpp_llama_tokenize(llama_context_ptr ctx, const char* text, bool add_special, bool parse_special /*false*/) {
  llama_context *c = static_cast<llama_context *>(ctx);
  std::string txt{text};

  std::vector<llama_token> ts = llama_tokenize(c, txt, add_special, parse_special);

  //   std::cout << ">>> C++ prompt: " << txt << "\n";
  //   std::cout << ">>> C++ Tokens:\n";
  //   for (int i = 0; i < ts.size(); ++i) {
  //     std::cout << ts[i] << " ";
  //   }
  //   std::cout << std::endl;

  Tokens tokens;
  tokens.len = ts.size();
  tokens.tokens = (int32_t *)malloc(tokens.len * sizeof(int32_t));
  memcpy(tokens.tokens, ts.data(), tokens.len * sizeof(int32_t));
  return tokens;
}

const char *lpp_llama_token_to_piece(llama_context_ptr ctx, llama_token token,
                                     bool special /*true*/) {
    llama_context *c = static_cast<llama_context *>(ctx);
    std::string piece = llama_token_to_piece(c,token, special);
    char *cstr = new char[piece.size() + 1];
    std::strcpy(cstr, piece.c_str());
    return cstr;
}

void lpp_llama_batch_add(llama_batch* batch, llama_token id, llama_pos pos, llama_seq_id* seq_ids, size_t seq_ids_len, bool logits) {
  if (seq_ids == nullptr) {
    llama_batch_add(*batch, id, pos, {0}, logits);
  } else {
    const std::vector<llama_seq_id> s_ids(seq_ids, seq_ids+seq_ids_len);
    llama_batch_add(*batch, id, pos, s_ids, logits);
  }
}

void lpp_llama_batch_clear(llama_batch *batch) { llama_batch_clear(*batch); }