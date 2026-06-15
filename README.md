MedSAM-Agent and SAMMED=3d adapted for each other



\# SAM-Med3D-Agent: Complete Training Blueprint

\## Goal

Train a multimodal LLM (Qwen3-VL) to act as an intelligent prompting agent for a frozen SAM-Med3D model. The MLLM looks at 3D medical volumes (represented as tri-view 2D slices), reasons about segmentation quality, and outputs 3D prompt coordinates to guide SAM-Med3D toward accurate volumetric segmentation — iteratively and autonomously.



\*\*SAM-Med3D is never trained. It is a frozen inference engine. Only the MLLM is trained.\*\*



\---



\## System Architecture Overview



```

\[3D Medical Volume]

&#x20;       ↓

\[Multi-View Interface] → extracts Axial, Coronal, Sagittal slices

&#x20;       ↓

\[MLLM - Qwen3-VL] ← sees tri-view image + text instruction

&#x20;       ↓

\[JSON Tool Call] → add\_3d\_point / add\_3d\_bbox / stop\_action

&#x20;       ↓

\[SAM-Med3D - FROZEN] → receives 3D prompt → outputs 3D mask

&#x20;       ↓

\[Updated Mask] → re-render tri-view → feed back to MLLM

&#x20;       ↓

\[Loop until stop\_action]

```



\---



\## Component 1: Multi-View Visual Interface



\### Purpose

Bridge the dimensionality gap. SAM-Med3D works in 3D. The MLLM only understands 2D images. This component translates the 3D world into something the MLLM can see.



\### Logic

\- Extract three orthogonal 2D slices from the 3D volume

\- Center the slices on the \*\*focus point\*\*:

&#x20; - Turn 0 (no mask yet): center of the volume OR center of provided bounding box

&#x20; - Turn 1+: centroid of the current predicted 3D mask

\- Overlay the current predicted mask in semi-transparent green on each slice

\- Stitch the three views side-by-side into one image: `\[Axial | Coronal | Sagittal]`

\- Feed the combined image to the MLLM as a single input



\---



\## Component 2: Action Space (JSON Tool Schema)



\### Purpose

Define the exact structured outputs the MLLM is allowed to produce. These map directly to SAM-Med3D's accepted input prompts.



\### Tools



```json

// Tool 1: Place a 3D point prompt

{

&#x20; "name": "add\_3d\_point",

&#x20; "arguments": {

&#x20;   "point\_3d": \[x, y, z],

&#x20;   "point\_type": "positive"  // or "negative"

&#x20; }

}



// Tool 2: Place a 3D bounding box prompt

{

&#x20; "name": "add\_3d\_bbox",

&#x20; "arguments": {

&#x20;   "bbox\_3d": \[x1, y1, z1, x2, y2, z2]

&#x20; }

}



// Tool 3: Terminate the loop

{

&#x20; "name": "stop\_action"

}

```



\### Coordinate System Rules

\- All coordinates are in \*\*voxel space\*\* matching the original volume dimensions `\[Z, Y, X]`

\- Points clicked on the 2D tri-view images must be \*\*mapped back to 3D voxel coordinates\*\* before being passed to SAM-Med3D

\- The mapping uses the known focus point and slice indices at the time of the click



\---



\## Component 3: Trajectory Dataset Generation



\### Purpose

Generate thousands of simulated expert interaction trajectories. This is the training data for the MLLM. SAM-Med3D runs the actual segmentation during simulation — the dataset records what the "expert" saw and what action it took at each turn.



\### Algorithm



```

For each labeled volume in the dataset:



&#x20; 1. INITIAL PROMPT

&#x20;    - Option A: Place a point at the centroid of the GT mask \[x, y, z]

&#x20;    - Option B: Place a bounding box around GT mask with random jitter ±5-10 voxels

&#x20;    - Render tri-view centered at volume center → save image\_t0



&#x20; 2. RUN SAM-MED3D

&#x20;    - Pass the prompt → receive predicted 3D mask

&#x20;    - Compute 3D IoU vs ground truth



&#x20; 3. ERROR ANALYSIS

&#x20;    - Find all False Negative voxels (GT=1, Pred=0) → missed regions

&#x20;    - Find all False Positive voxels (GT=0, Pred=1) → over-segmented regions

&#x20;    - Apply 3D distance transform to find centroid of largest error cluster

&#x20;    - Assign point type: FN cluster → positive point, FP cluster → negative point



&#x20; 4. PROGRESS CHECK

&#x20;    - Place new point → run SAM-Med3D → compute new 3D IoU

&#x20;    - If ΔIoU > τ (threshold = 0.04): accept the click, continue

&#x20;    - If ΔIoU ≤ τ: retry at next largest error cluster

&#x20;    - Render updated tri-view centered on new mask centroid → save image\_tN



&#x20; 5. TERMINATION

&#x20;    - Stop when: IoU converges, max turns reached (e.g., 5), or IoU > 0.90

&#x20;    - Final action recorded as stop\_action



&#x20; 6. STORE TRAJECTORY

&#x20;    {

&#x20;      "volume\_id": "case\_042",

&#x20;      "target\_structure": "liver",

&#x20;      "turns": \[

&#x20;        {

&#x20;          "turn": 0,

&#x20;          "image\_path": "trajectories/case\_042/turn\_0.png",

&#x20;          "instruction": "Segment the liver",

&#x20;          "action": {

&#x20;            "name": "add\_3d\_point",

&#x20;            "arguments": {"point\_3d": \[144, 112, 64], "point\_type": "positive"}

&#x20;          },

&#x20;          "iou\_after": 0.71

&#x20;        },

&#x20;        {

&#x20;          "turn": 1,

&#x20;          "image\_path": "trajectories/case\_042/turn\_1.png",

&#x20;          "instruction": "Segment the liver",

&#x20;          "action": {

&#x20;            "name": "add\_3d\_point",

&#x20;            "arguments": {"point\_3d": \[150, 120, 50], "point\_type": "positive"}

&#x20;          },

&#x20;          "iou\_after": 0.87

&#x20;        },

&#x20;        {

&#x20;          "turn": 2,

&#x20;          "image\_path": "trajectories/case\_042/turn\_2.png",

&#x20;          "instruction": "Segment the liver",

&#x20;          "action": {"name": "stop\_action"},

&#x20;          "iou\_after": 0.87

&#x20;        }

&#x20;      ]

&#x20;    }

```



\### Target Dataset Size

\- Minimum: 10,000 trajectories across diverse structures and modalities

\- Recommended: 50,000+ for robust generalization



\---



\## Component 4: Two-Phase MLLM Training



\### Phase 1 — Supervised Fine-Tuning (SFT)



| Item | Detail |

|---|---|

| \*\*Base Model\*\* | Qwen3-VL |

| \*\*Input\*\* | Tri-view PNG + conversation history + text instruction |

| \*\*Output\*\* | JSON tool call string |

| \*\*Loss\*\* | Cross-entropy (next token prediction) |

| \*\*Framework\*\* | LLaMA-Factory or HuggingFace TRL |

| \*\*Data Format\*\* | ChatML / ShareGPT conversation format |

| \*\*Goal\*\* | MLLM learns to imitate the simulated expert trajectories |



\*\*Example training sample (ChatML format):\*\*

```json

{

&#x20; "messages": \[

&#x20;   {

&#x20;     "role": "system",

&#x20;     "content": "You are a 3D medical segmentation agent. You are looking at three orthogonal slices (Axial, Coronal, Sagittal) from a 3D medical volume stitched left-to-right. The current segmentation mask is shown in semi-transparent green. Output the correct tool call to improve the segmentation."

&#x20;   },

&#x20;   {

&#x20;     "role": "user",

&#x20;     "content": \[

&#x20;       {"type": "image", "image": "trajectories/case\_042/turn\_0.png"},

&#x20;       {"type": "text", "text": "Segment the liver"}

&#x20;     ]

&#x20;   },

&#x20;   {

&#x20;     "role": "assistant",

&#x20;     "content": "<tool\_call>{\\"name\\": \\"add\_3d\_point\\", \\"arguments\\": {\\"point\_3d\\": \[144, 112, 64], \\"point\_type\\": \\"positive\\"}}</tool\_call>"

&#x20;   },

&#x20;   {

&#x20;     "role": "user",

&#x20;     "content": \[

&#x20;       {"type": "image", "image": "trajectories/case\_042/turn\_1.png"},

&#x20;       {"type": "text", "text": "Segment the liver"}

&#x20;     ]

&#x20;   },

&#x20;   {

&#x20;     "role": "assistant",

&#x20;     "content": "<tool\_call>{\\"name\\": \\"stop\_action\\"}</tool\_call>"

&#x20;   }

&#x20; ]

}

```



\---



\### Phase 2 — Reinforcement Learning with Verifiable Rewards (GRPO)



| Item | Detail |

|---|---|

| \*\*Algorithm\*\* | GRPO (Group Relative Policy Optimization) |

| \*\*Environment\*\* | Live SAM-Med3D inference in the training loop |

| \*\*Reward Signal\*\* | Computed from 3D mask quality vs ground truth |

| \*\*Goal\*\* | MLLM learns to exceed imitation — find better strategies |



\*\*Reward Function:\*\*



| Reward Component | Value | Trigger |

|---|---|---|

| Format Reward | +0.5 | Correct JSON tool schema output |

| Quality Reward | 3D Dice + 3D IoU (weighted sum) | After each SAM-Med3D inference |

| Improvement Bonus | +0.2 per turn | Each turn where IoU strictly increases |

| Overshoot Penalty | -0.3 | Any turn where IoU decreases |

| Tool-Cost Penalty | -0.05 per extra turn | Every turn beyond turn 1 |

| Stop Bonus | +0.3 | Calling stop\_action at peak IoU |



\---



\## Training Definition of Done



> The MLLM reliably guides SAM-Med3D to a \*\*3D Dice Score above target threshold\*\* on the held-out validation set, using \*\*3 or fewer interaction turns\*\* on average.



Once this bar is hit, the agentic loop (navigation tools, scroll actions, upstream detection) is added on top.



\---



\## What Is NOT Being Built Yet

\- ❌ Navigation / scroll tools

\- ❌ MIP-based blind search

\- ❌ Full autonomous clinical deployment loop

\- ❌ Any modification to SAM-Med3D weights



\---



\## Immediate Next Steps



1\. Confirm base dataset (modality + target structures)

2\. Confirm base MLLM (Qwen3-VL recommended)

3\. Set up SAM-Med3D inference pipeline (frozen)

4\. Build and run trajectory generation script

5\. Convert trajectories to ChatML format

6\. Run Phase 1 SFT

7\. Evaluate → if plateau, move to Phase 2 GRPO

