#!/usr/bin/env python
"""Prior-guided inference for a trained HiFTS checkpoint."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from hifts.prompts import SYSTEM_PROMPT, system_prompt_with_prior


def predict_global_prior(essay_text, prior_tokenizer, prior_model, device: str) -> float:
    encoded = prior_tokenizer(
        essay_text, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = prior_model(**encoded).logits
    if logits.shape[-1] == 1:
        return float(logits.squeeze().item())
    probs = torch.softmax(logits, dim=-1)
    score_values = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
    return float((probs * score_values).sum(dim=-1).item())


def generate(
    essay_text: str,
    llm_path: str,
    prior_model_path: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_prior: bool,
) -> str:
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True).to(device)
    llm.eval()

    if use_prior:
        prior_tokenizer = AutoTokenizer.from_pretrained(prior_model_path, trust_remote_code=True)
        prior_model = AutoModelForSequenceClassification.from_pretrained(
            prior_model_path, trust_remote_code=True
        ).to(device)
        prior_model.eval()
        prior_score = predict_global_prior(essay_text, prior_tokenizer, prior_model, device)
        system_prompt = system_prompt_with_prior(prior_score)
    else:
        system_prompt = SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": essay_text},
    ]
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = llm.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
    gen_ids = output_ids[0][model_inputs["input_ids"].shape[1] :]
    return llm_tokenizer.decode(gen_ids, skip_special_tokens=True)


def parse_args():
    p = argparse.ArgumentParser(description="HiFTS inference with optional BERT holistic prior.")
    p.add_argument("--llm_path", default=os.environ.get("LLM_PATH", str(ROOT / "checkpoints/qwen3_grpo/final_model")))
    p.add_argument(
        "--prior_model_path",
        default=os.environ.get("PRIOR_MODEL_PATH", str(ROOT / "checkpoints/bert_holistic_prior")),
    )
    p.add_argument("--essay_text", default=os.environ.get("ESSAY_TEXT", ""))
    p.add_argument("--essay_file", default="")
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    p.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "512")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.6")))
    p.add_argument("--top_p", type=float, default=float(os.environ.get("TOP_P", "0.9")))
    p.add_argument("--no_prior", action="store_true", help="Disable the holistic BERT prior.")
    return p.parse_args()


def main():
    args = parse_args()
    essay = args.essay_text
    if args.essay_file:
        essay = Path(args.essay_file).read_text(encoding="utf-8")
    if not essay.strip():
        raise ValueError("Provide --essay_text or --essay_file.")
    print(
        generate(
            essay_text=essay,
            llm_path=args.llm_path,
            prior_model_path=args.prior_model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_prior=not args.no_prior,
        )
    )


if __name__ == "__main__":
    main()
