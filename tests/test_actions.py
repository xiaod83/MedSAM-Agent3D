"""Standard-library regression tests runnable without ML dependencies."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location('standalone_' + name, ROOT / 'medsam_agent3d' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

parsing = load('parsing')
clicker = load('clicker')

class ActionValidationTests(unittest.TestCase):
    def test_garbage_does_not_stop(self):
        action = parsing.parse_action('I cannot produce JSON')
        self.assertEqual(action['name'], 'invalid_action')
        self.assertIn('_error', action)

    def test_invalid_arguments_are_rejected(self):
        for coords in ([1, 2], [1, 2, 1000], [1, 2, float('nan')], [True, 2, 3], [1.5, 2, 3]):
            action = parsing.validate_action({'name': 'add_3d_point', 'arguments': {'point_3d': coords, 'point_type': 'positive'}})
            self.assertEqual(action['name'], 'invalid_action')
        self.assertEqual(parsing.validate_action({'name': 'stop_action', 'arguments': None})['name'], 'invalid_action')

    def test_wrong_json_types_are_rejected(self):
        for payload in ('{"name": []}', '{"name": {}}', '{"name":"add_3d_point","arguments":{"point_3d":[1,2,3],"point_type":[]}}'):
            self.assertEqual(parsing.parse_action(payload)['name'], 'invalid_action')
        self.assertEqual(parsing.parse_action(None)['name'], 'invalid_action')

    def test_string_braces_and_escaped_quotes(self):
        text = '{"name":"stop_action","note":"a } and \\\" quote"}'
        self.assertEqual(parsing.parse_action(text)['name'], 'stop_action')

    def test_legacy_and_navigation_actions(self):
        action = parsing.parse_action('{"name":"add_3d_point","point_3d":[0,500,999],"point_type":"positive"}')
        self.assertEqual(action['arguments']['point_3d'], [0, 500, 999])
        self.assertEqual(parsing.denormalize_point((0, 500, 999), (9, 11, 13)), (0, 5, 12))
        self.assertEqual(parsing.validate_action({'name': 'navigate', 'arguments': {'point_3d': [1, 2, 3]}})['name'], 'navigate')

    def test_panel_points_validate_view_and_coordinates(self):
        good = {'name': 'add_view_point', 'arguments': {'view': 'axial', 'point_2d': [180, 350], 'point_type': 'positive'}}
        self.assertEqual(parsing.validate_action(good)['name'], 'add_view_point')
        for changes in ({'view': []}, {'view': 'unknown'}, {'point_2d': [1, 2, 3]}, {'point_2d': [10**400, 0]}, {'point_type': 'other'}):
            bad = {'name': good['name'], 'arguments': dict(good['arguments'], **changes)}
            self.assertEqual(parsing.validate_action(bad)['name'], 'invalid_action')

    def test_prompt_has_no_center_click_example(self):
        prompt = load('prompting').build_system_prompt((179, 192, 294), ['femur_left'])
        self.assertIn('add_view_point', prompt)
        self.assertIn('femur_left', prompt)
        self.assertNotIn('[500, 500, 500]', prompt)
        self.assertIn('not a suggested target', prompt)

    def test_native_qwen_xml_and_reported_malformed_enums(self):
        for body in (
            '<function=add_view_point><parameter=view>axial</parameter><parameter=point_2d>[100,350]</parameter><parameter=point_type>positive</parameter></function>',
            '<function=add_view_point><function=add_view_point><parameter=view="axial</parameter><parameter=point_2d>\n[100, 350]</parameter><parameter=point_type="positive</parameter></function>',
        ):
            text = '<tool_call>' + body + '</tool_call>'
            action = parsing.parse_action(text)
            self.assertEqual(action['name'], 'add_view_point', action)
            self.assertEqual(action['arguments'], {'view': 'axial', 'point_2d': [100,350], 'point_type': 'positive'})
            self.assertEqual(action['_raw_output'], text)

    def test_native_xml_rejects_ambiguous_or_invalid_calls(self):
        for body in (
            '<function=stop_action></function><function=stop_action></function>',
            '<function=add_view_point><parameter=view>axial</parameter><parameter=view>coronal</parameter></function>',
            '<function=add_view_point><parameter=point_2d>[1,1000]</parameter></function>',
            '<function=stop_action>',
            '<function=stop_action><parameter=x>__import__("os")</parameter></function>',
        ):
            self.assertEqual(parsing.parse_action('<tool_call>'+body+'</tool_call>')['name'], 'invalid_action')
        self.assertEqual(parsing.parse_action('<tool_call><function=stop_action></function></tool_call>')['name'], 'stop_action')

    def test_click_chronology_survives_round_trip(self):
        tracker = clicker.Clicker3D()
        tracker.add_negative((1, 2, 3), turn=0)
        tracker.add_positive((4, 5, 6), turn=1)
        self.assertEqual(tracker.get_last_click().coords, (4, 5, 6))
        restored = clicker.Clicker3D()
        restored.set_state(tracker.get_state())
        self.assertEqual(restored.get_all_labels(), [0, 1])
        self.assertEqual(restored.get_last_click().turn, 1)

if __name__ == '__main__':
    unittest.main()
