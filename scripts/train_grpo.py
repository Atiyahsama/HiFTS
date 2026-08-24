#!/usr/bin/env python
"""GRPO training for HiFTS.

Paper defaults (Sec. 5): Qwen3-4B SFT checkpoint, G=4, β=0.02, lr=1e-6, α=0.8.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from hifts.rewards import (
    CustomRewardFunction,
    group_normalize_rewards,
    load_and_process_data,
    reward_lambdas_from_alpha,
)
from hifts.traits import TRAIT_COLUMNS, load_trait_map


@dataclass
class Config:
    data_path: str = os.environ.get("DATA_PATH", str(ROOT / "CFMS-34/train.csv"))
    model_path: str = os.environ.get("MODEL_PATH", str(ROOT / "checkpoints/qwen3_sft"))
    output_dir: str = os.environ.get("OUTPUT_DIR", str(ROOT / "checkpoints/qwen3_grpo"))
    reward_model_path: str = os.environ.get("REWARD_MODEL_PATH", str(ROOT / "models/bge-small-zh"))
    main_device: str = os.environ.get("MAIN_DEVICE", "cuda:0")
    ref_device: str = os.environ.get("REF_DEVICE", "cuda:0")
    max_length: int = int(os.environ.get("MAX_LENGTH", "1024"))
    max_new_tokens: int = int(os.environ.get("MAX_NEW_TOKENS", "512"))
    epochs: int = int(os.environ.get("EPOCHS", "2"))
    batch_size: int = int(os.environ.get("BATCH_SIZE", "1"))
    group_size: int = int(os.environ.get("GROUP_SIZE", "4"))
    lr: float = float(os.environ.get("LR", "1e-6"))
    kl_coef: float = float(os.environ.get("KL_COEF", "0.02"))
    save_steps: int = int(os.environ.get("SAVE_STEPS", "10"))
    do_sample: bool = os.environ.get("DO_SAMPLE", "1") == "1"
    temperature: float = float(os.environ.get("TEMPERATURE", "0.6"))
    top_p: float = float(os.environ.get("TOP_P", "0.9"))
    alpha: float = float(os.environ.get("ALPHA", "0.8"))


class RewardModule(torch.nn.Module):
    def __init__(self, reward_fn):
        super().__init__()
        self.reward_fn = reward_fn

    def forward(self, responses, gts):
        vals = [self.reward_fn.compute_reward(r, g) for r, g in zip(responses, gts)]
        return torch.tensor(vals, dtype=torch.float32)


def build_collator(tokenizer):
    def collate(batch):
        ids = [torch.tensor(item["input_ids"]) for item in batch]
        masks = [torch.tensor(item["attention_mask"]) for item in batch]
        gts = [item["ground_truth_response"] for item in batch]
        return {
            "input_ids": pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id),
            "attention_mask": pad_sequence(masks, batch_first=True, padding_value=0),
            "ground_truth_response": gts,
        }

    return collate


def main():
    cfg = Config()
    if os.environ.get("GRPO_OUTPUT_APPEND_ALPHA", "1") != "0":
        cfg.output_dir = f"{cfg.output_dir}_alpha{cfg.alpha:g}"
    os.makedirs(cfg.output_dir, exist_ok=True)
    lambdas = reward_lambdas_from_alpha(cfg.alpha)
    print(f"REWARD_ALPHA={cfg.alpha}")
    print(f"REWARD_LAMBDAS={lambdas}")

    mapping = load_trait_map()
    print(f"TRAIT_MAP={mapping}")
    print(f"CORE_TRAITS={TRAIT_COLUMNS}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    df = pd.read_csv(cfg.data_path)
    train_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    dataset = load_and_process_data(train_df, tokenizer)

    def tokenize(sample):
        encoded = tokenizer(sample["prompt"], truncation=True, max_length=cfg.max_length)
        encoded["ground_truth_response"] = sample["ground_truth_response"]
        return encoded

    dataset = dataset.map(tokenize, batched=False)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=build_collator(tokenizer),
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    policy = AutoPeftModelForCausalLM.from_pretrained(
        cfg.model_path,
        quantization_config=bnb,
        trust_remote_code=True,
        is_trainable=True,
    ).to(cfg.main_device)
    policy.config.use_cache = False
    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()
    policy.generation_config = GenerationConfig(eos_token_id=tokenizer.eos_token_id)

    ref_policy = AutoPeftModelForCausalLM.from_pretrained(
        cfg.model_path,
        quantization_config=bnb,
        trust_remote_code=True,
        is_trainable=False,
    ).to(cfg.ref_device)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    reward_module = RewardModule(
        CustomRewardFunction(
            alpha=cfg.alpha,
            embedding_model=cfg.reward_model_path,
            device=cfg.main_device,
            lambdas=lambdas,
        )
    )
    generation_kwargs = {
        "do_sample": cfg.do_sample,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": 0,
        "max_new_tokens": cfg.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    global_step = 0
    for _ in range(cfg.epochs):
        for batch in dataloader:
            policy.train()
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(cfg.main_device)
            attention_mask = batch["attention_mask"].to(cfg.main_device)
            ground_truths = batch["ground_truth_response"]

            for i in range(input_ids.size(0)):
                prompt_ids = input_ids[i].unsqueeze(0)
                prompt_mask = attention_mask[i].unsqueeze(0)
                gt = ground_truths[i]
                sampled_ids = []
                rewards = []
                for _ in range(cfg.group_size):
                    seq = policy.generate(
                        input_ids=prompt_ids, attention_mask=prompt_mask, **generation_kwargs
                    )
                    prompt_len = int(prompt_mask.sum().item())
                    gen_ids = seq[0][prompt_len:]
                    sampled_ids.append(gen_ids)
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    rewards.append(reward_module([text], [gt])[0])

                rewards = torch.stack(rewards).to(cfg.main_device)
                advantages = group_normalize_rewards(rewards).detach()

                for j in range(cfg.group_size):
                    full = torch.cat([prompt_ids[0], sampled_ids[j]]).unsqueeze(0)
                    logits = policy(full).logits
                    resp_len = sampled_ids[j].shape[0]
                    resp_logits = logits[:, -resp_len - 1 : -1]
                    resp_targets = sampled_ids[j].unsqueeze(0)
                    log_probs = torch.log_softmax(resp_logits, dim=-1)
                    token_logprob = torch.gather(log_probs, 2, resp_targets.unsqueeze(-1)).squeeze(-1)
                    seq_logprob = token_logprob.sum()

                    with torch.no_grad():
                        ref_logits = ref_policy(full.to(cfg.ref_device)).logits[:, -resp_len - 1 : -1]
                        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                        ref_token_logprob = torch.gather(
                            ref_log_probs,
                            2,
                            resp_targets.to(cfg.ref_device).unsqueeze(-1),
                        ).squeeze(-1)
                        ref_seq_logprob = ref_token_logprob.sum().to(cfg.main_device)

                    kl = seq_logprob - ref_seq_logprob
                    loss = -advantages[j] * seq_logprob + cfg.kl_coef * kl
                    (loss / cfg.group_size).backward()

            optimizer.step()
            global_step += 1
            if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                save_dir = os.path.join(cfg.output_dir, f"step_{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                policy.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

    final_dir = os.path.join(cfg.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
