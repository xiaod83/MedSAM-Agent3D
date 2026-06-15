import os
import json


def format_trajectories_for_sft(trajectories_dir, output_jsonl_path):
    """
    Converts trajectory JSONs into multi-turn ChatML format for SFT.
    Each JSONL sample contains ALL turns of a trajectory as a single
    conversation, so the MLLM learns sequential decision-making.

    Compatible with Qwen3-VL's apply_chat_template.
    """
    system_prompt = (
        "You are a 3D medical segmentation agent. You are looking at three "
        "orthogonal slices (Axial, Coronal, Sagittal) from a 3D medical volume "
        "stitched left-to-right. The current segmentation mask is shown in "
        "semi-transparent green. Output the correct tool call to improve the "
        "segmentation."
    )

    formatted_data = []

    for filename in os.listdir(trajectories_dir):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(trajectories_dir, filename)
        with open(file_path, 'r') as f:
            trajectory = json.load(f)

        # Collect all images and build a multi-turn conversation
        turn_keys = sorted(trajectory.keys(), key=lambda k: int(k.split('_')[1]))
        images = []
        messages = [{"role": "system", "content": system_prompt}]

        for turn_key in turn_keys:
            turn_data = trajectory[turn_key]
            image_path = turn_data.get("image", "").replace("\\", "/")
            instruction = turn_data.get("instruction", "")
            action = turn_data.get("action", {})
            action_str = json.dumps(action)

            images.append(image_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": f"<tool_call>{action_str}</tool_call>"
            })

        sample = {"images": images, "conversations": messages}
        formatted_data.append(sample)

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Formatted {len(formatted_data)} multi-turn trajectories into {output_jsonl_path}")


if __name__ == "__main__":
    import sys
    format_trajectories_for_sft(
        sys.argv[1] if len(sys.argv) > 1 else "data/trajectories",
        sys.argv[2] if len(sys.argv) > 2 else "data/sft_dataset.jsonl"
    )
