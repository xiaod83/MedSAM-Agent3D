import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments
from trl import SFTTrainer

class MedicalTrajectoryDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.data = []
        # Explicitly enforce UTF-8 to prevent Windows charmap codec errors
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["images"][0]
        
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        # Format the text using chat template
        text = self.processor.apply_chat_template(
            item["conversations"],
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {
            "image": image,
            "text": text,
            "conversations": item["conversations"]
        }

def custom_collate_fn(batch, processor, max_seq_length=8192):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    
    # Tokenize with truncation to prevent OOM
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )
    
    # Clone input_ids for labels
    labels = inputs["input_ids"].clone()
    
    # Find special token IDs for Qwen3-VL's chat template
    # Pattern: last occurrence of <|im_start|>assistant
    im_start_id = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
    assistant_ids = processor.tokenizer.encode('assistant', add_special_tokens=False)
    
    for i in range(labels.shape[0]):
        ids = inputs["input_ids"][i]
        
        # Find the last occurrence of [im_start_id] + assistant_ids
        last_assistant_start = -1
        for j in range(len(ids) - len(assistant_ids)):
            if ids[j].item() == im_start_id:
                match = True
                for k, aid in enumerate(assistant_ids):
                    if j + 1 + k >= len(ids) or ids[j + 1 + k].item() != aid:
                        match = False
                        break
                if match:
                    last_assistant_start = j
        
        if last_assistant_start >= 0:
            # Mask everything up to and including the assistant header
            labels[i, :last_assistant_start] = -100

    # Mask out the padding tokens from loss
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    
    return inputs

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_sft(dataset_path="data/sft_dataset.jsonl", model_id="Qwen/Qwen3-VL-8B-Instruct", output_dir="models/sft_output", seed=42, **model_kwargs):
    """
    Fine-tune Qwen3-VL on the SFT dataset.
    
    Args:
        dataset_path: Path to the JSONL dataset
        model_id: HuggingFace model ID or local path
        output_dir: Where to save the fine-tuned model
        seed: Random seed
        **model_kwargs: Additional kwargs passed to AutoModelForImageTextToText.from_pretrained()
                       (e.g., quantization_config for 4-bit QLoRA on low-memory GPUs)
    """
    set_seed(seed)
    print(f"Loading processor and model {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Merge default kwargs with any overrides
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    load_kwargs.update(model_kwargs)
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        **load_kwargs
    )

    dataset = MedicalTrajectoryDataset(dataset_path, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Reduced to 1 for VLM memory safety
        gradient_accumulation_steps=8,
        warmup_steps=100,
        max_steps=1000,
        learning_rate=2e-5,
        bf16=True, # Qwen3-VL prefers bfloat16
        logging_steps=10,
        save_steps=200,
        remove_unused_columns=False,
    )

    print("Initializing Multimodal SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=lambda batch: custom_collate_fn(batch, processor),
        max_seq_length=8192,  # Increased for image tokens (Qwen3-VL uses ~256+ tokens per image)
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    train_sft(dataset_path=args.dataset, model_id=args.model, output_dir=args.output, seed=args.seed)