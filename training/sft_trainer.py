import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments
from trl import SFTTrainer


class MedicalTrajectoryDataset(Dataset):
    """Dataset for multi-turn trajectories with multiple images per sample."""

    def __init__(self, jsonl_path, processor):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_paths = item["images"]
        conversations = item["conversations"]

        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in image_paths]

        text = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )

        return {"images": images, "text": text, "conversations": conversations}


def custom_collate_fn(batch, processor, max_seq_length=8192):
    """Collate function handling multi-turn (multiple images per sample)."""
    all_images = [img for item in batch for img in item["images"]]
    texts = [item["text"] for item in batch]

    inputs = processor(
        text=texts,
        images=all_images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )

    labels = inputs["input_ids"].clone()

    # Mask prompt tokens: keep loss ONLY on assistant response tokens.
    # Multi-turn conversations have multiple assistant spans — unmask ALL of them.
    im_start_id = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    assistant_ids = processor.tokenizer.encode('assistant', add_special_tokens=False)

    for i in range(labels.shape[0]):
        ids = inputs["input_ids"][i]
        # Start by masking everything
        labels[i, :] = -100

        # Find ALL assistant response spans and unmask them
        j = 0
        while j < len(ids) - len(assistant_ids) - 1:
            if ids[j].item() == im_start_id:
                match = all(
                    ids[j + 1 + k].item() == aid
                    for k, aid in enumerate(assistant_ids)
                    if j + 1 + k < len(ids)
                )
                if match:
                    # Found <|im_start|>assistant — start of assistant response
                    start = j
                    # Find the next <|im_end|> or <|im_start|>
                    for end in range(j + len(assistant_ids), len(ids)):
                        if ids[end].item() == im_end_id:
                            labels[i, start:end+1] = ids[start:end+1]
                            j = end
                            break
                        if ids[end].item() == im_start_id:
                            # Next turn started — assistant didn't end cleanly
                            labels[i, start:end] = ids[start:end]
                            j = end
                            break
                    else:
                        # Reached end of sequence
                        labels[i, start:] = ids[start:]
                        break
                    continue
            j += 1

    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_sft(dataset_path="data/sft_dataset.jsonl",
              model_id="Qwen/Qwen3-VL-8B-Instruct",
              output_dir="models/sft_output",
              seed=42,
              **model_kwargs):
    set_seed(seed)
    print(f"Loading processor and model {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    load_kwargs.update(model_kwargs)

    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    dataset = MedicalTrajectoryDataset(dataset_path, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        max_steps=1000,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=lambda batch: custom_collate_fn(batch, processor),
        max_seq_length=8192,
    )

    print("Starting Training...")
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/sft_dataset.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", default="models/sft_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_sft(dataset_path=args.dataset, model_id=args.model,
              output_dir=args.output, seed=args.seed)
