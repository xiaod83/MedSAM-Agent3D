"""
Vision-language agent wrapper for Qwen3-VL and native-vision Qwen3.5/3.6/3.8.

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
    """Wraps Qwen3-VL and native-vision Qwen3.5/3.6/3.8 for tool calls.

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
        import transformers
        from transformers import AutoConfig, AutoProcessor

        config_kwargs = {key: load_kwargs[key] for key in
                         ('revision', 'token', 'cache_dir', 'local_files_only', 'subfolder')
                         if key in load_kwargs}
        try:
            config = AutoConfig.from_pretrained(model_id, **config_kwargs)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Cannot load model configuration for {model_id}. Native-vision "
                "Qwen3.5/3.6/3.8 requires a recent Transformers 5.x release "
                "supporting qwen3_5; upgrade the remote runtime dependencies."
            ) from exc
        self._direct_vision_processor = config.model_type in {'qwen3_5', 'qwen3_5_moe'}
        class_names = {
            'qwen3_vl': 'Qwen3VLForConditionalGeneration',
            'qwen3_vl_moe': 'Qwen3VLMoeForConditionalGeneration',
            'qwen3_5': 'Qwen3_5ForConditionalGeneration',
            'qwen3_5_moe': 'Qwen3_5MoeForConditionalGeneration',
        }
        class_name = class_names.get(config.model_type)
        if class_name is None:
            raise ValueError(f"Unsupported vision model architecture: {config.model_type}. "
                             "Choose a Qwen3-VL or native-vision Qwen3.5/3.6/3.8 checkpoint.")
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            raise ImportError(f"Transformers {transformers.__version__} lacks {class_name}; "
                              "install a recent Transformers 5.x release for this checkpoint.")

        self.model_id = model_id
        print(f"Loading {model_id} ({config.model_type})...")
        defaults = dict(
            torch_dtype=(torch.float32 if device_map == "cpu" or not torch.cuda.is_available()
                         else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
            device_map=device_map,
            trust_remote_code=True,
        )
        defaults.update(load_kwargs)

        if self._direct_vision_processor:
            dtype = defaults.pop('torch_dtype')
            defaults.setdefault('dtype', dtype)
        self.model = model_class.from_pretrained(model_id, config=config, **defaults)
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, **config_kwargs
        )
        self.model.eval()
        print(f"{model_id} loaded on {self.model.device}")

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

        if self._direct_vision_processor:
            from .prompting import TOOL_DEFINITIONS
            # Let the checkpoint's own template supply its native tool grammar.
            # Remove the legacy JSON-only footer to avoid conflicting instructions.
            messages[0]['content'] = system_prompt.split('Return exactly one JSON tool call', 1)[0]
            # Native-vision Qwen3.5/3.6/3.8 processors handle the PIL image and
            # multimodal tokenization together. Disable default thinking so a
            # bounded tool-call budget is not consumed by unfinished reasoning.
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
                tools=TOOL_DEFINITIONS,
            ).to(self.model.device)
        else:
            from qwen_vl_utils import process_vision_info
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                return_tensors="pt", padding=True,
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
