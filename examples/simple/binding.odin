package llama_odin

import "core:c"


when ODIN_OS == .Linux do foreign import lpp "../../llama.cpp/binding.a"


gpt_params_ptr :: distinct rawptr
llama_model_params_ptr :: distinct rawptr
llama_model_ptr :: distinct rawptr
llama_context_params_ptr :: distinct rawptr
llama_context_ptr :: distinct rawptr
llama_sampler_ptr :: distinct rawptr

llama_token :: distinct c.int32_t
llama_pos :: distinct c.int32_t
llama_seq_id :: distinct c.int32_t

llama_sampler_chain_params :: struct {
	no_perf: c.bool,
}

// Input data for llama_decode
// A llama_batch object can contain input about one or many sequences
// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
//
// - token  : the token ids of the input (used when embd is NULL)
// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
// - pos    : the positions of the respective token in the sequence
// - seq_id : the sequence to which the respective token belongs
// - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
//
llama_batch :: struct {
	n_tokens:   c.int32_t,
	token:      [^]llama_token,
	embd:       [^]c.float,
	pos:        [^]llama_pos,
	n_seq_id:   [^]c.int32_t,
	seq_id:     ^^llama_seq_id, // TODO: should it be [^][^]llama_seq_id?
	logits:     [^]c.int8_t,
	all_pos_0:  llama_pos, // used if pos == NULL
	all_pos_1:  llama_pos, // used if pos == NULL
	all_seq_id: llama_seq_id, // used if seq_id == NULL
}

Tokens :: struct {
	len:    c.int32_t,
	tokens: [^]llama_token,
}


llama_example :: enum c.int {
	LLAMA_EXAMPLE_COMMON,
	LLAMA_EXAMPLE_SPECULATIVE,
	LLAMA_EXAMPLE_MAIN,
	LLAMA_EXAMPLE_INFILL,
	LLAMA_EXAMPLE_EMBEDDING,
	LLAMA_EXAMPLE_PERPLEXITY,
	LLAMA_EXAMPLE_RETRIEVAL,
	LLAMA_EXAMPLE_PASSKEY,
	LLAMA_EXAMPLE_IMATRIX,
	LLAMA_EXAMPLE_BENCH,
	LLAMA_EXAMPLE_SERVER,
	LLAMA_EXAMPLE_CVECTOR_GENERATOR,
	LLAMA_EXAMPLE_EXPORT_LORA,
	LLAMA_EXAMPLE_LLAVA,
	LLAMA_EXAMPLE_LOOKUP,
	LLAMA_EXAMPLE_PARALLEL,
	LLAMA_EXAMPLE_COUNT,
}

// numa strategies
ggml_numa_strategy :: enum c.int {
	GGML_NUMA_STRATEGY_DISABLED = 0,
	GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
	GGML_NUMA_STRATEGY_ISOLATE = 2,
	GGML_NUMA_STRATEGY_NUMACTL = 3,
	GGML_NUMA_STRATEGY_MIRROR = 4,
	GGML_NUMA_STRATEGY_COUNT,
}

llama_perf_type :: enum c.int {
	LLAMA_PERF_TYPE_CONTEXT       = 0,
	LLAMA_PERF_TYPE_SAMPLER_CHAIN = 1,
}


@(default_calling_convention = "c")
foreign lpp {
	lpp_print_usage :: proc(argc: c.int, argv: ^^c.char) ---

	lpp_make_gpt_params_ptr :: proc() -> gpt_params_ptr ---

	lpp_gpt_params_parse :: proc(argc: c.int, argv: ^^c.char, params: gpt_params_ptr, ex: llama_example) -> c.bool ---

	lpp_get_n_predict :: proc(params: gpt_params_ptr) -> c.int32_t ---
	lpp_get_n_gpu_layers :: proc(params: gpt_params_ptr) -> c.int32_t ---
	lpp_get_model :: proc(params: gpt_params_ptr) -> cstring ---
	lpp_get_prompt :: proc(params: gpt_params_ptr) -> cstring ---

	llama_backend_init :: proc() ---
	llama_backend_free :: proc() ---

	lpp_get_numa :: proc(params: gpt_params_ptr) -> ggml_numa_strategy ---

	llama_numa_init :: proc(numa: ggml_numa_strategy) ---

	lpp_llama_model_params_from_gpt_params :: proc(params: gpt_params_ptr) -> llama_model_params_ptr ---

	lpp_llama_load_model_from_file :: proc(model_path: cstring, model_params: llama_model_params_ptr) -> llama_model_ptr ---

	llama_free_model :: proc(model: llama_model_ptr) ---

	lpp_llama_context_params_from_gpt_params :: proc(params: gpt_params_ptr) -> llama_context_params_ptr ---

	llp_llama_new_context_with_model :: proc(model: llama_model_ptr, ctx_params: llama_context_params_ptr) -> llama_context_ptr ---

	llama_sampler_chain_default_params :: proc() -> llama_sampler_chain_params ---

	llama_sampler_chain_init :: proc(params: llama_sampler_chain_params) -> llama_sampler_ptr ---

	llama_free :: proc(ctx: llama_context_ptr) ---

	llama_sampler_free :: proc(smpl: llama_sampler_ptr) ---

	llama_sampler_init_greedy :: proc() -> llama_sampler_ptr ---
	llama_sampler_chain_add :: proc(chain: llama_sampler_ptr, smpl: llama_sampler_ptr) ---

	lpp_llama_tokenize :: proc(ctx: llama_context_ptr, text: cstring, add_special: c.bool,  /*false*/parse_special: c.bool) -> Tokens ---

	lpp_llama_token_to_piece :: proc(ctx: llama_context_ptr, token: llama_token,  /*true*/special: c.bool) -> cstring ---

	llama_batch_init :: proc(n_tokens_alloc: c.int32_t, embd: c.int32_t, n_seq_max: c.int32_t) -> llama_batch ---
	llama_batch_free :: proc(batch: llama_batch) ---

	lpp_llama_batch_add :: proc(batch: ^llama_batch, id: llama_token, pos: llama_pos, seq_ids: [^]llama_seq_id, seq_ids_len: c.size_t, logits: c.bool) ---

	llama_decode :: proc(ctx: llama_context_ptr, batch: llama_batch) -> c.int32_t ---

	ggml_time_us :: proc() -> c.int64_t ---

	llama_n_ctx :: proc(ctx: llama_context_ptr) -> c.int ---

	llama_sampler_sample :: proc(smpl: llama_sampler_ptr, ctx: llama_context_ptr, idx: c.int32_t) -> llama_token ---

	llama_token_is_eog :: proc(model: llama_model_ptr, token: llama_token) -> c.bool ---

	lpp_llama_batch_clear :: proc(batch: ^llama_batch) ---

	llama_perf_print :: proc(ptr: rawptr, perf_type: llama_perf_type) ---
}
