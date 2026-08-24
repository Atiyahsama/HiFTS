#!/usr/bin/env python
"""PPO training for HiFTS (ablation / comparison with GRPO)."""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from hifts.rewards import CustomRewardFunction, load_and_process_data, reward_lambdas_from_alpha
from hifts.traits import TRAIT_COLUMNS, load_trait_map


@dataclass
class Config:
    data_path: str = os.environ.get("DATA_PATH", str(ROOT / "CFMS-34/train.csv"))
    model_path: str = os.environ.get("MODEL_PATH", str(ROOT / "checkpoints/qwen3_sft"))
    output_dir: str = os.environ.get("OUTPUT_DIR", str(ROOT / "checkpoints/qwen3_ppo"))
    reward_model_path: str = os.environ.get("REWARD_MODEL_PATH", str(ROOT / "models/bge-small-zh"))
    device: str = os.environ.get("DEVICE", "cuda:0")
    max_length: int = int(os.environ.get("MAX_LENGTH", "1024"))
    max_new_tokens: int = int(os.environ.get("MAX_NEW_TOKENS", "512"))
    batch_size: int = int(os.environ.get("BATCH_SIZE", "4"))
    mini_batch_size: int = int(os.environ.get("MINI_BATCH_SIZE", "1"))
    lr: float = float(os.environ.get("LR", "1e-6"))
    ppo_epochs: int = int(os.environ.get("PPO_EPOCHS", "10"))
    save_steps: int = int(os.environ.get("SAVE_STEPS", "100"))
    alpha: float = float(os.environ.get("ALPHA", "0.8"))
    temperature: float = float(os.environ.get("TEMPERATURE", "0.6"))
    top_p: float = float(os.environ.get("TOP_P", "0.9"))


class RewardModule(torch.nn.Module):
    def __init__(self, reward_fn):
        super().__init__()
        self.reward_fn = reward_fn

    def forward(self, responses, gts):
        values = [self.reward_fn.compute_reward(r, g) for r, g in zip(responses, gts)]
        return torch.tensor(values, dtype=torch.float32)


def build_collator(tokenizer):
    def collate(batch):
        gts = [item["ground_truth_response"] for item in batch]
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "ground_truth_response": gts,
        }

    return collate


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    lambdas = reward_lambdas_from_alpha(cfg.alpha)
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

    def tokenize(example):
        tok = tokenizer(example["prompt"], truncation=True, max_length=cfg.max_length)
        tok["ground_truth_response"] = example["ground_truth_response"]
        return tok

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
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.model_path, quantization_config=bnb, trust_remote_code=True
    ).to(cfg.device)
    policy.config.use_cache = False
    policy.enable_input_require_grads()
    policy.generation_config = GenerationConfig(eos_token_id=tokenizer.eos_token_id)

    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, quantization_config=bnb, trust_remote_code=True
    ).to(cfg.device)
    ref_model.eval()

    reward = RewardModule(
        CustomRewardFunction(
            alpha=cfg.alpha,
            embedding_model=cfg.reward_model_path,
            device=cfg.device,
            lambdas=lambdas,
        )
    )
    ppo_config = PPOConfig(
        exp_name="hifts-ppo",
        seed=40,
        learning_rate=cfg.lr,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        gradient_accumulation_steps=1,
        ppo_epochs=cfg.ppo_epochs,
        init_kl_coef=0.02,
        adap_kl_ctrl=True,
    )
    trainer = PPOTrainer(config=ppo_config, model=policy, ref_model=ref_model, tokenizer=tokenizer)
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": cfg.top_p,
        "do_sample": True,
        "temperature": cfg.temperature,
        "max_new_tokens": cfg.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    global_step = 0
    for _ in range(cfg.ppo_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            gts = batch["ground_truth_response"]
            queries = [input_ids[i] for i in range(input_ids.size(0))]
            gen = policy.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_scores=False,
                **generation_kwargs,
            )
            responses = []
            for i in range(input_ids.size(0)):
                prompt_len = int(attention_mask[i].sum().item())
                responses.append(gen.sequences[i][prompt_len:])
            texts = tokenizer.batch_decode(responses, skip_special_tokens=True)
            rewards = reward(texts, gts).to(cfg.device)
            trainer.step(queries=queries, responses=responses, scores=list(rewards.unbind(0)))
            global_step += 1
            if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                save_dir = os.path.join(cfg.output_dir, f"step_{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                trainer.model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

    final_dir = os.path.join(cfg.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
