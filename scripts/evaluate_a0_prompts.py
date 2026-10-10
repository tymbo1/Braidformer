#!/usr/bin/env python3
"""Run a small fixed-prompt qualitative evaluation from the A0 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from braidformer.config import load_config
from braidformer.models import A0TransformerLM
from braidformer.training.integrity import sha256_file, set_reproducible_seed, verify_checkpoint


FIXED_PROMPTS = (
    "Once upon a time, a small fox found a shiny key.",
    "Lucy wanted to help her brother build a tall tower.",
    "On Saturday morning, Sam and his dog went to the garden.",
    "A little bird was afraid to leave its nest, but then",
)


def generate_greedy(
    model: A0TransformerLM,
    tokenizer: object,
    prompt: str,
    *,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, object]:
    """Generate a fixed-length-or-EOS continuation without sampling."""

    encoded = tokenizer.encode(prompt, add_special_tokens=False)  # type: ignore[attr-defined]
    prompt_ids = list(encoded.ids)
    input_ids = torch.tensor([[bos_id, *prompt_ids]], dtype=torch.long, device=device)
    generated_ids: list[int] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids).logits[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            generated_ids.append(next_id)
            input_ids = torch.cat(
                (input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)),
                dim=1,
            )
            if next_id == eos_id:
                break
    full_ids = input_ids[0].tolist()[1:]
    return {
        "prompt": prompt,
        "generated_token_count": len(generated_ids),
        "generated_token_ids": generated_ids,
        "text": tokenizer.decode(full_ids, skip_special_tokens=True),  # type: ignore[attr-defined]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "braidnano_a0_baseline_proposal.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "results" / "runs" / "a0_seed17_pilot" / "checkpoint_step_001000.pt",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "a0_prompt_eval.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is unavailable")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    checkpoint_record = verify_checkpoint(checkpoint_path)
    if checkpoint_record.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("checkpoint config hash does not match evaluation config")

    data_config = json.loads(
        (REPO_ROOT / config.data_and_tokenizer).read_text(encoding="utf-8")
    )
    tokenizer_path = REPO_ROOT / data_config["tokenizer"]["artifact"]
    tokenizer_sha256 = sha256_file(tokenizer_path)
    if tokenizer_sha256 != data_config["prepared_artifacts"]["tokenizer_sha256"]:
        raise RuntimeError("approved tokenizer hash mismatch")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise RuntimeError("approved tokenizer vocabulary mismatch")
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("approved tokenizer is missing BOS/EOS ids")

    set_reproducible_seed(config.seed, deterministic=True)
    device = torch.device("cuda")
    model = A0TransformerLM(config)
    if model.parameter_count() != 41_866_240:
        raise RuntimeError("unexpected A0 parameter count")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    generations = [
        generate_greedy(
            model,
            tokenizer,
            prompt,
            bos_id=bos_id,
            eos_id=eos_id,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        for prompt in FIXED_PROMPTS
    ]
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "status": "completed",
        "evaluation": "fixed_prompt_greedy_generation",
        "training_started": False,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "checkpoint_step": checkpoint_record["step"],
        "tokenizer_sha256": tokenizer_sha256,
        "seed": config.seed,
        "decoding": {
            "method": "greedy_argmax",
            "max_new_tokens": args.max_new_tokens,
            "sampling": False,
        },
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "elapsed_seconds": elapsed,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "generations": generations,
        "interpretation": "qualitative fixed-prompt inspection only; not a capability benchmark",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
