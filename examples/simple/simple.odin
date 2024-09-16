package llama_odin

import "core:c"
import "core:c/libc"
import "core:fmt"
import "core:os"
import "core:strings"

main :: proc() {
	args := os.args
	defer delete(args)

	argc := len(args)
	argv := make([dynamic]cstring, argc)
	defer delete(argv)

	for i in 0 ..< argc {
		argv[i] = strings.unsafe_string_to_cstring(args[i])
	}

	params := lpp_make_gpt_params_ptr()
	defer libc.free(params)
	ok := lpp_gpt_params_parse(
		cast(c.int)argc,
		cast(^^c.char)(&argv[0]),
		params,
		llama_example.LLAMA_EXAMPLE_COMMON,
	)

	fmt.printfln("Parsing params: %v", ok)
	if !ok {
		fmt.eprintln("ERROR: failed to parse CLI arguments. Aborting.")
		os.exit(1)
	}
	fmt.printfln(
		`Some of the params:
    tokens to predict: %d
    GPU layers: %d
    model name: %s
    prompt: %s`,
		lpp_get_n_predict(params),
		lpp_get_n_gpu_layers(params),
		lpp_get_model(params),
		lpp_get_prompt(params),
	)

	// total length of the sequence including the prompt
	n_predict := lpp_get_n_predict(params)

	// init LLM

	llama_backend_init()
	defer llama_backend_free()

	numa := lpp_get_numa(params)
	llama_numa_init(numa)

	// initialize the model
	model_params: llama_model_params_ptr = lpp_llama_model_params_from_gpt_params(params)
	defer libc.free(model_params)

	model: llama_model_ptr = lpp_llama_load_model_from_file(lpp_get_model(params), model_params)
	defer llama_free_model(model)
	if model == nil {
		fmt.eprintln("ERROR: unable to load model. Aborting.")
		os.exit(1)
	}

	// initialize the context

	ctx_params: llama_context_params_ptr = lpp_llama_context_params_from_gpt_params(params)
	defer libc.free(ctx_params)

	ctx: llama_context_ptr = llp_llama_new_context_with_model(model, ctx_params)
	defer llama_free(ctx)
	if ctx == nil {
		fmt.eprintln("ERROR: unable to initialize context. Aborting.")
		os.exit(1)
	}

	sparams: llama_sampler_chain_params = llama_sampler_chain_default_params()
	sparams.no_perf = false

	smpl: llama_sampler_ptr = llama_sampler_chain_init(sparams)
	defer llama_sampler_free(smpl)

	llama_sampler_chain_add(smpl, llama_sampler_init_greedy())

	// tokenize the prompt
	tokens_list: Tokens = lpp_llama_tokenize(ctx, lpp_get_prompt(params), true, false)
	defer libc.free(tokens_list.tokens)
	fmt.println(">>> Odin Tokens:")
	for i in 0 ..< tokens_list.len {
		fmt.printf("%d ", tokens_list.tokens[i])
	}
	fmt.println()

	n_ctx: c.int = llama_n_ctx(ctx)
	n_kv_req: c.int = tokens_list.len + (n_predict - tokens_list.len)
	fmt.printfln(">>> n_predict = %d, n_ctx = %d, n_kv_req = %d", n_predict, n_ctx, n_kv_req)

	// make sure the KV cache is big enough to hold all the prompt and generated tokens
	if (n_kv_req > n_ctx) {
		fmt.eprintln("ERROR: n_kv_req > n_ctx, the required KV cache size is not big enough")
		fmt.eprintln("       either reduce n_predict or increase n_ctx")
		os.exit(1)
	}

	// print the prompt token-by-token
	fmt.eprintln(">>> Print the prompt token-by-token:")
	for i in 0 ..< tokens_list.len {
		t := tokens_list.tokens[i]
		tkn: cstring = lpp_llama_token_to_piece(ctx, t, true)
		fmt.eprint("%s", tkn)
	}
	fmt.eprintln()

	// create a llama_batch with size 512
	// we use this object to submit token data for decoding

	batch: llama_batch = llama_batch_init(512, 0, 1)
	defer llama_batch_free(batch)

	// evaluate the initial prompt
	for i in 0 ..< tokens_list.len {
		lpp_llama_batch_add(&batch, tokens_list.tokens[i], cast(llama_pos)i, nil, 0, false)
	}

	// llama_decode will output logits only for the last token of the prompt
	batch.logits[batch.n_tokens - 1] = 1 // true

	if llama_decode(ctx, batch) != 0 {
		fmt.eprintln("ERROR: llama_decode() failed. Aborting")
		os.exit(1)
	}

	// main loop

	n_cur := batch.n_tokens
	n_decode := 0

	t_main_start := ggml_time_us()

	for n_cur <= n_predict {
		// sample the next token
		{
			new_token_id: llama_token = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1)

			// is it an end of generation?
			if (llama_token_is_eog(model, new_token_id) || n_cur == n_predict) {
				fmt.print("\n")
				break
			}

			fmt.printf("%s", lpp_llama_token_to_piece(ctx, new_token_id, true))

			// prepare the next batch
			lpp_llama_batch_clear(&batch)

			// push this new token for next evaluation
			lpp_llama_batch_add(&batch, new_token_id, cast(llama_pos)n_cur, nil, 0, true)

			n_decode += 1
		}

		n_cur += 1

		// evaluate the current batch with the transformer model
		if llama_decode(ctx, batch) != 0 {
			fmt.eprintfln("ERROR: failed to eval, return code %d. Aborting", 1)
			os.exit(1)
		}
	}
	fmt.print("\n")

	t_main_end := ggml_time_us()

	fmt.printfln(
		"decoded %d tokens in %.2f s, speed: %.2f t/s\n",
		n_decode,
		cast(f32)(t_main_end - t_main_start) / 1000000.0,
		cast(f32)n_decode / (cast(f32)(t_main_end - t_main_start) / 1000000.0),
	)

	llama_perf_print(smpl, llama_perf_type.LLAMA_PERF_TYPE_SAMPLER_CHAIN)
	llama_perf_print(ctx, llama_perf_type.LLAMA_PERF_TYPE_CONTEXT)

}
