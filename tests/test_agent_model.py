"""Model routing and multimodal preprocessing without downloads or ML packages."""
import contextlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class Inputs(dict):
    def to(self, device):
        self.device = device
        return self


class AgentModelTests(unittest.TestCase):
    def setUp(self):
        torch = types.ModuleType('torch')
        torch.float32, torch.float16, torch.bfloat16 = 'float32', 'float16', 'bfloat16'
        torch.cuda = types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)
        torch.no_grad = contextlib.nullcontext
        pil = types.ModuleType('PIL')
        pil.Image = types.SimpleNamespace(Image=object)
        package = types.ModuleType('_agent_regression')
        package.__path__ = [str(ROOT / 'medsam_agent3d')]
        self.transformers = types.ModuleType('transformers')
        self.transformers.__version__ = '5.test'
        self.transformers.AutoConfig = Mock()
        self.transformers.AutoProcessor = Mock()
        self.model = Mock(device='cuda:0')
        # Generated IDs support the same batch/token slicing as a tensor.
        class Generated:
            def __getitem__(self, index):
                return 'generated-only-ids'
        self.model.generate.return_value = Generated()
        self.native_class, self.legacy_class = Mock(), Mock()
        self.native_class.from_pretrained.return_value = self.model
        self.legacy_class.from_pretrained.return_value = self.model
        self.transformers.Qwen3_5ForConditionalGeneration = self.native_class
        self.transformers.Qwen3VLForConditionalGeneration = self.legacy_class
        self.inputs = Inputs(input_ids=types.SimpleNamespace(shape=(1, 12)), pixel_values='image-tensor')
        self.processor = Mock()
        self.processor.apply_chat_template.return_value = self.inputs
        self.processor.return_value = self.inputs
        self.processor.tokenizer.pad_token_id = 0
        self.processor.decode.return_value = '{"name":"add_3d_point","arguments":{"point_3d":[1,2,3],"point_type":"positive"}}'
        self.transformers.AutoProcessor.from_pretrained.return_value = self.processor
        self.vision = types.ModuleType('qwen_vl_utils')
        self.vision.process_vision_info = Mock(return_value=(['image'], None))
        self.modules = patch.dict(sys.modules, {'torch': torch, 'PIL': pil,
            'transformers': self.transformers, 'qwen_vl_utils': self.vision,
            '_agent_regression': package})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location('_agent_regression.agent', ROOT / 'medsam_agent3d/agent.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.Agent = module.QwenVLAgent

    def configure(self, model_type):
        self.transformers.AutoConfig.from_pretrained.return_value = types.SimpleNamespace(model_type=model_type)

    def test_native_vision_routing_disables_thinking_and_processes_image_directly(self):
        self.configure('qwen3_5')
        agent = self.Agent('Qwen/Qwen3.8-27B')
        image = object()
        action = agent.generate_action(image, 'segment', 'system', temperature=0)
        self.native_class.from_pretrained.assert_called_once()
        self.legacy_class.from_pretrained.assert_not_called()
        call = self.processor.apply_chat_template.call_args
        self.assertIs(call.args[0][-1]['content'][0]['image'], image)
        self.assertEqual(call.kwargs, dict(tokenize=True, add_generation_prompt=True,
                         return_dict=True, return_tensors='pt', enable_thinking=False,
                         tools=call.kwargs['tools']))
        self.assertIn('add_view_point', [t['function']['name'] for t in call.kwargs['tools']])
        self.vision.process_vision_info.assert_not_called()
        self.processor.assert_not_called()
        self.assertEqual(action['name'], 'add_3d_point')
        self.assertEqual(self.model.generate.call_args.kwargs['pixel_values'], 'image-tensor')
        self.assertFalse(self.model.generate.call_args.kwargs['do_sample'])

    def test_original_qwen3vl_uses_existing_vision_utility(self):
        self.configure('qwen3_vl')
        self.processor.apply_chat_template.return_value = 'formatted-chat'
        agent = self.Agent('Qwen/Qwen3-VL-8B-Instruct')
        agent.generate_action(object(), 'segment', 'system')
        self.legacy_class.from_pretrained.assert_called_once()
        self.native_class.from_pretrained.assert_not_called()
        self.vision.process_vision_info.assert_called_once()
        self.assertEqual(self.vision.process_vision_info.call_args.kwargs['image_patch_size'], 16)
        self.processor.assert_called_once()

    def test_unsupported_text_only_architecture_fails_before_weight_load(self):
        self.configure('qwen3')
        with self.assertRaisesRegex(ValueError, 'Unsupported vision model architecture'):
            self.Agent('text-only')
        self.native_class.from_pretrained.assert_not_called()

    def test_old_transformers_reports_required_architecture(self):
        self.configure('qwen3_5')
        del self.transformers.Qwen3_5ForConditionalGeneration
        with self.assertRaisesRegex(ImportError, 'recent Transformers 5.x'):
            self.Agent('Qwen/Qwen3.8-27B')


if __name__ == '__main__':
    unittest.main()
