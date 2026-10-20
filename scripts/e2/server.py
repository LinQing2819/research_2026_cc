"""vLLM 0.8.5 launcher with read-only runtime identity/tokenization endpoints."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from memgate.experiments.common import file_hash, require, tree_hashes, versions


def main():
    import uvloop
    import vllm
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.utils import cli_env_setup
    from vllm.utils import FlexibleArgumentParser

    require(vllm.__version__ == "0.8.5", "This runtime bridge is verified against vLLM 0.8.5 only")
    cli_env_setup()
    parser = make_arg_parser(FlexibleArgumentParser(description=__doc__))
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    require(Path(args.model).is_dir(), "E2 requires a local, hashable model directory")
    require(args.tokenizer is None or Path(args.tokenizer).resolve() == Path(args.model).resolve(),
            "E2 tokenizer must come from the same local model directory")
    print("Hashing server model files for E2 identity...", flush=True)
    files = tree_hashes(args.model)
    original_build, original_init = api_server.build_app, api_server.init_app_state

    def build_app(parsed_args, *extra, **kw):
        app = original_build(parsed_args, *extra, **kw)

        @app.get("/e2/identity")
        async def identity():
            return app.state.e2_identity

        @app.post("/e2/chat_ids")
        async def chat_ids(body: dict):
            # Use the tokenizer exposed by the actual live engine, not a second loader.
            tokenizer = await app.state.engine_client.get_tokenizer()
            messages = body["messages"]
            options = dict(add_generation_prompt=True, enable_thinking=False)
            return dict(ids=tokenizer.apply_chat_template(messages, tokenize=True, **options),
                        text=tokenizer.apply_chat_template(messages, tokenize=False, **options))

        return app

    async def init_state(engine_client, vllm_config, state, parsed_args):
        await original_init(engine_client, vllm_config, state, parsed_args)
        import torch
        model = vllm_config.model_config
        state.e2_identity = dict(
            actual_max_model_len=model.max_model_len,
            served_models=parsed_args.served_model_name or [parsed_args.model],
            model_files=files, dtype=str(model.dtype),
            tensor_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            enforce_eager=model.enforce_eager,
            generation_config=model.generation_config,
            hf_config=model.hf_config.to_dict(),
            cuda_version=torch.version.cuda,
            gpu=[dict(name=torch.cuda.get_device_properties(i).name,
                      total_memory=torch.cuda.get_device_properties(i).total_memory,
                      capability=list(torch.cuda.get_device_capability(i)))
                 for i in range(torch.cuda.device_count())],
            runtime_versions=versions(["vllm", "transformers", "tokenizers", "torch"]),
            launcher_sha256=file_hash(__file__),
            argv=sys.argv[1:],
        )

    api_server.build_app, api_server.init_app_state = build_app, init_state
    uvloop.run(api_server.run_server(args))


if __name__ == "__main__":
    main()
