import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import json
import re
import numpy as np

# NOTE: The README specifies GRPO (Group Relative Policy Optimization) for Phase 2,
# NOT PPO with a value head. GRPO samples multiple responses per prompt, computes
# advantages relative to the group mean (no critic network), and updates the policy
# via group-relative normalization.
#
# This module implements a simplified PPO-style loop for prototyping. The reward
# function matches the README spec (format + quality + improvement + overshoot penalty
# + tool cost + stop bonus) and can be swapped into a GRPO framework.
#
# For production GRPO, consider:
# - LLaMA-Factory (supports GRPO natively)
# - OpenRLHF (supports VLMs + GRPO)
# - VERL (supports multi-turn RL)

def parse_action(text_output):
    """
    Extract JSON action from MLLM output.
    Uses a targeted regex to find the action JSON block without
    greedily matching across multiple objects.
    """
    try:
        patterns = [
            r'\{\s*"name"\s*:\s*"add_3d_point"\s*,\s*"point_3d"\s*:\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\s*,\s*"type"\s*:\s*"(?:positive|negative)"\s*\}',
            r'\{\s*"name"\s*:\s*"stop_action"\s*\}',
            r'\{[^{}]*\}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_output, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                if "name" in result:
                    return result
    except (json.JSONDecodeError, AttributeError):
        pass
    
    return {"name": "stop_action"}


class VLValueHead(torch.nn.Module):
    """
    A lightweight value head that operates on the VLM's last hidden state.
    Attaches after the language model backbone to predict state values for PPO.
    """
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.value_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states):
        return self.value_head(hidden_states).squeeze(-1)


def train_rl(sft_model_path="models/sft_output", output_dir="models/rl_output"):
    """
    Custom RL training loop for VLM-based segmentation agent.
    
    NOTE: This is a simplified reference implementation. For production use,
    integrate with a VLM-compatible RL framework.
    """
    print(f"Loading SFT multimodal model and processor from {sft_model_path}...")
    
    processor = AutoProcessor.from_pretrained(sft_model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        sft_model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Attach value head
    hidden_size = model.config.hidden_size
    value_head = VLValueHead(hidden_size).to(model.device)
    
    # Optimizer: only train value head + LoRA parameters (not the vision encoder)
    optimizer = torch.optim.AdamW(
        list(value_head.parameters()) + list(model.lm_head.parameters()),
        lr=1.41e-5
    )
    
    # Environment setup — uses real TotalSegmentator volumes from the SFT dataset
    from training.rl_environment import MedSAMEnvironment
    from models.sam_med3d_wrapper import FrozenSAMMed3D
    from data_generation.totalsegmentator_loader import get_totalsegmentator_dataset, load_nifti

    ts_dir = os.path.join(os.path.dirname(__file__), '..', 'Totalsegmentator_dataset_v201')
    dataset = get_totalsegmentator_dataset(ts_dir)
    if not dataset:
        print("No TotalSegmentator data found for RL env. Skipping environment setup.")
        return

    sam_model = FrozenSAMMed3D()
    sample = dataset[0]
    vol, spacing = load_nifti(sample['ct_path'])
    mask, _ = load_nifti(sample['mask_path'])
    mask = (mask > 0).astype(np.uint8)
    env = MedSAMEnvironment(sam_model, vol, mask, sample['target_name'], sample['volume_id'], spacing)
    print(f"RL env initialized on {sample['volume_id']}/{sample['target_name']}")
    
    print("Starting custom VLM-compatible RL rollout loop...")
    
    for epoch in range(1):
        state = env.reset()
        
        # 1. Prepare inputs using HuggingFace messages format
        messages = [
            {"role": "system", "content": "You are a 3D medical segmentation agent. You see three orthogonal slices [Axial | Coronal | Sagittal]. Output the correct tool call."},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": state['instruction']}
            ]}
        ]
        
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text_prompt],
            images=[state['image']],
            return_tensors="pt"
        ).to(model.device)
        
        # 2. Generate action
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
        
        # Extract only the new tokens
        prompt_len = inputs['input_ids'].shape[1]
        response_ids = generated_ids[0, prompt_len:]
        response_text = processor.decode(response_ids, skip_special_tokens=True)
        
        # 3. Parse action and step environment
        action_json = parse_action(response_text)
        action_point = action_json.get("point_3d", state['center'])
        click_type = action_json.get("type", "positive")
        
        next_state, reward, done = env.step(tuple(action_point), click_type)
        print(f"Action: {action_point} ({click_type}), Reward: {reward:.4f}")
        
        # 4. Compute value and PPO loss (simplified)
        # In a full implementation, we would:
        #   a. Forward pass through the VLM to get logits + hidden states
        #   b. Compute log-probabilities of the chosen action tokens
        #   c. Compute advantage = reward - value (for single-step)
        #   d. Compute PPO clipped surrogate loss
        #   e. Backward pass + optimizer step
        
        # For now, compute a reference loss on the action generation
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # last token hidden state
            value = value_head(last_hidden).item()
        
        print(f"Estimated state value: {value:.4f}")
        print(f"Advantage (reward - value): {reward - value:.4f}")
        
    print("RL implementation completed. Save the model:")
    print(f"  model.save_pretrained('{output_dir}')")
    print(f"  processor.save_pretrained('{output_dir}')")
    # Uncomment to save:
    # model.save_pretrained(output_dir)
    # processor.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sft_output")
    parser.add_argument("--output", default="models/rl_output")
    args = parser.parse_args()
    
    train_rl(sft_model_path=args.model, output_dir=args.output)
