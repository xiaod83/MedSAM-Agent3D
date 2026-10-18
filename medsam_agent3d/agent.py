"""
MLLM agent wrapper for Qwen3-VL.

Provides a clean interface for the vision-language model used as the
"reasoning brain" in the agentic loop. The agent sees tri-view images
and generates tool-call actions in JSON format.

References:
    MedSAM-Agent grounding_model abstraction -- same role, separate object.
    Qwen3-VL chat template with process_vision_info.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from .parsing import parse_action


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class MLLMAgent(ABC):
    """Abstract interface for any multimodal LLM used as the grounding agent.

    This allows swapping Qwen3-VL for another model (LLaVA, GPT-4V, etc.)
    without changing the agent loop.
    """

    @abstractmethod
    def generate_action(
        self,
        image: Image.Image,
        instruction: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate the next action given the current tri-view image and instruction.

        Args:
            image: Tri-view PIL Image (Axial | Coronal | Sagittal).
            instruction: Text instruction (e.g., "Segment the liver").
            system_prompt: Full system prompt with tool definitions.
            conversation_history: Prior turns as [{"role": "...", "content": "..."}, ...].
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).

        Returns:
            Action dict with keys: name, arguments, _raw_output.
        """
        ...


# ---------------------------------------------------------------------------
# Qwen3-VL implementation
# ---------------------------------------------------------------------------

class QwenVLAgent(MLLMAgent):
    """Wraps Qwen3-VL for tool-call generation (no fine-tuning needed).

    Handles:
    - Chat template application with vision inputs
    - Image processing via qwen_vl_utils.process_vision_info
    - Text generation and action parsing
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device_map: str = "auto",
        **load_kwargs,
    ):
        """Load Qwen3-VL model and processor.

        Args:
            model_id: HuggingFace model ID or local path.
            device_map: Device map ('auto', 'cuda', 'cpu').
            **load_kwargs: Additional kwargs for from_pretrained
                (e.g., quantization_config, torch_dtype).
        """
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_id = model_id
        print(f"Loading Qwen3-VL from {model_id}...")
        defaults = dict(
            torch_dtype=(torch.float32 if device_map == "cpu" or not torch.cuda.is_available()
                         else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
            device_map=device_map,
            trust_remote_code=True,
        )
        defaults.update(load_kwargs)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, **defaults
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model.eval()
        print(f"Qwen3-VL loaded on {self.model.device}")

    def generate_action(
        self,
        image: Image.Image,
        instruction: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate the next tool-call action from the MLLM.

        Args:
            image: Tri-view PIL Image.
            instruction: Text instruction.
            system_prompt: Full system prompt.
            conversation_history: Prior turns.
            max_new_tokens: Max generation length.
            temperature: Sampling temperature.

        Returns:
            Parsed action dict with name, arguments, _raw_output.
        """
        from qwen_vl_utils import process_vision_info

        # --- Build messages ---
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        })

        # --- Apply chat template ---
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # --- Process vision inputs (proper Qwen3-VL image handling) ---
        image_inputs, video_inputs = process_vision_info(
            messages, image_patch_size=16
        )

        # --- Tokenize ---
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        # --- Generate ---
        do_sample = temperature > 0.0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_p": 0.9,
            })

        with torch.no_grad():
            generated = self.model.generate(**inputs, **generation_kwargs)

        # --- Decode only new tokens ---
        prompt_len = inputs["input_ids"].shape[1]
        response_ids = generated[0, prompt_len:]
        response_text = self.processor.decode(
            response_ids, skip_special_tokens=True
        ).strip()

        return parse_action(response_text)
