#### MedSAM-Agent: Empowering Interactive Medical Image Segmentation with
Multi-turn Agentic Reinforcement Learning
Shengyuan Liu 1 Liuxin Bao 1 Qi Yang 2 3 Wanting Geng 2 4 Boyun Zheng 1
Chenxin Li 1 Wenting Chen 5 Houwen Peng 2 † Yixuan Yuan 1 †
Abstract
Medical image segmentation is evolving from
task-specific models toward generalizable frame-
works. Recent research leverages Multi-modal
```
Large Language Models (MLLMs) as au-
```
tonomous agents, employing reinforcement learn-
```
ing with verifiable reward (RLVR) to orches-
```
trate specialized tools like the Segment Anything
```
Model (SAM). However, these approaches often
```
rely on single-turn, rigid interaction strategies
and lack process-level supervision during train-
ing, which hinders their ability to fully exploit the
dynamic potential of interactive tools and leads
to redundant actions. To bridge this gap, we pro-
pose MedSAM-Agent, a framework that refor-
mulates interactive segmentation as a multi-step
autonomous decision-making process. First, we
introduce a hybrid prompting strategy for expert-
curated trajectory generation, enabling the model
to internalize human-like decision heuristics and
adaptive refinement strategies. Furthermore, we
develop a two-stage training pipeline that inte-
grates multi-turn, end-to-end outcome verifica-
tion with a clinical-fidelity process reward design
to promote interaction parsimony and decision
efficiency. Extensive experiments across 6 med-
ical modalities and 21 datasets demonstrate that
MedSAM-Agent achieves state-of-the-art perfor-
mance, effectively unifying autonomous medi-
cal reasoning with robust, iterative optimization.
Code is available here.
1Chinese University of Hong Kong, Hong Kong SAR, China
2Hunyuan Group, Tencent 3Institute of Automation, the Chinese
Academy of Sciences, Beijing, China 4Dalian University of Tech-
nology, Dalian, China 5Stanford University, Stanford, USA. Corre-
spondence to: Yixuan Yuan <yxyuan@ee.cuhk.edu.hk>, Houwen
Peng <henryllpeng@tencent.com>.
SAM
Point/Box
Prompt
Seg
DecoderMLLM
<seg>
token
Doctor
Multi-turn Refinement❌
Full-Automated✅
Multi-turn Refinement
❌ Full-Automated
✅
```
(a) SAM-based models (SAM, MedSAM)
```
```
(b) MLLM-driven models (LISA, UniBioMed)
```
```
(c) MedSAM-Agent (Ours)
```
Multi-turn Refinement
Full-Automated✅
✅ SAMAction
Feedback
🔥
❄
❄Visual
Agent
Figure 1. Comparison of medical image segmentation paradigms.
```
(a) SAM-based models (e.g., SAM, MedSAM) require continuous
```
```
manual prompting via points or bounding boxes. (b) MLLM-
```
```
driven models (e.g., LISA, UniBioMed) employs MLLM with
```
```
specialized seg decoders and <seg> tokens. (c) Ours MedSAM-
```
Agent functions as an autonomous visual agent that performs multi-
turn refinement through iterative feedback and tool interaction,
emulating the professional decision-making process.
1. Introduction
Medical image segmentation stands as a foundational task
in clinical computer vision, underpinning critical applica-
tions such as early disease diagnosis, surgical planning, and
treatment response assessment. Traditional AI-driven seg-
```
mentation approaches, including UNet (Ronneberger et al.,
```
```
2015) and its variants (Chen et al., 2021; Isensee et al.,
```
```
2021; Li et al., 2025a; Zhou et al., 2018), have demon-
```
strated impressive performance in medical segmentation
scenarios. However, these methods are primarily tailored
to specific tasks or imaging modalities, posing significant
challenges when generalizing to new tasks that were not
encountered during model training. The emergence of the
```
Segment Anything Model (SAM) (Kirillov et al., 2023;
```
```
Ravi et al., 2024) represented a pivotal breakthrough, en-
```
abling high-quality segmentation through interactive human
```
prompts (e.g., points, bounding boxes, or masks) (Ma et al.,
```
```
2024; Wei et al., 2024b; Konwer et al., 2025; Ma et al.,
```
```
2025b; Wu et al., 2025a; Gong et al., 2024). Nevertheless,
```
1
```
arXiv:2602.03320v1 [cs.CV] 3 Feb 2026
```
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
these SAM-derived interactive models remain inherently
```
dependent on expert prompting (Fig. 1a), which prevents
```
them from achieving autonomous, generalized segmentation
without manual intervention.
Meanwhile, Multi-modal Large Language Models
```
(MLLMs) (Sellergren et al., 2025; Xu et al., 2025b; Jiang
```
```
et al., 2025a; Chen et al., 2024a) have shown remarkable
```
perception and reasoning abilities in the medical domain,
```
especially in visual question answering (Hu et al., 2024;
```
```
Ye et al., 2024; Liu et al., 2021; Lau et al., 2018; He
```
```
et al., 2020; Liu et al., 2025a) and report generation (Li
```
```
et al., 2025b; Zambrano Chaves et al., 2025; Ma et al.,
```
```
2025a; Chen et al., 2024b; Ru et al., 2026) tasks. While
```
```
early attempts (Wu et al., 2025b; Huang et al., 2025b;
```
```
Wang et al., 2025a) integrate MLLMs for segmentation
```
```
via implicit tokens and additional pixel decoders (Fig. 1b),
```
these methods often compromise semantic generalization
by altering the original output space and shifting away from
```
language-based outputs. Recent advancements (Li et al.,
```
```
2024; Nath et al., 2025; Fathi et al., 2025) have shifted
```
toward employing MLLMs as agents for invoking SAM
tools, utilizing Reinforcement Learning from Verifiable
```
Rewards (RLVR) during the post-training stage to integrate
```
the high-level reasoning of MLLMs with the robust
interactive segmentation of SAM by internalizing the
autonomous tool-using.
However, existing methods still suffer from two major limi-
```
tations: First, interaction strategies remain simplistic and in-
```
efficient. Current studies either rely on single-turn prompts
```
(You & Wu, 2025; Liu et al., 2025b; Huang et al., 2025a;
```
```
Li et al., 2024), treating SAM as a static segmentor rather
```
than an iterative agent, or adhere to rigid, point-only trajec-
```
tories (Zhu et al., 2025; Jiang et al., 2026). These point-
```
centric paradigms lack the spatial flexibility to adaptively
encompass morphological heterogeneity. In voluminous or
ambiguously bounded cases, such methods fail to leverage
bounding boxes as a human-like anchor for global con-
text, leading to suboptimal step-wise refinement and under-
utilized iterative potential. Second, the reinforcement learn-
ing process lacks process-level supervision. Existing RLVR
```
frameworks (You & Wu, 2025; Huang et al., 2025a; Liu
```
```
et al., 2025b; Jiang et al., 2025b; Liu et al., 2025c; Su et al.,
```
```
2025a) focus predominantly on terminal outcomes, such as
```
```
the final segmentation accuracy (e.g., Dice or IoU), while
```
neglecting the efficiency and logical coherence of interme-
diate actions. Without specific rewards to incentivize action
parsimony and per-step effectiveness, the agent may take
unnecessary steps that do not contribute to mask refinement,
which ultimately compromises both training convergence
efficiency and final inference performance.
To address these limitations, we propose MedSAM-Agent,
a framework that enables MLLMs to emulate human anno-
tators by treating the use of interactive segmentation tools
```
as a multi-step decision-making process (Fig. 1c). First,
```
we implement a hybrid prompting strategy to generate di-
verse, high-quality expert trajectories, enabling the model
to internalize human-like decision heuristics and adaptive
refinement strategies by boxes and points. Then we de-
velop a two-stage training pipeline that integrates multi-turn,
end-to-end outcome verification with a clinical-fidelity pro-
cess reward design to promote interaction parsimony and
decision efficiency. This framework jointly optimizes for
segmentation accuracy and procedural efficiency, ensuring
the agent learns to make precise, non-redundant decisions.
Consequently, MedSAM-Agent achieves state-of-the-art
performance across 6 medical modalities and 21 datasets,
demonstrating exceptional cross-modal generalization and
tool-use versatility. Our key contributions are summarized
as follows:
• We introduce the MedSAM-Agent framework, which
reformulates medical image segmentation by transition-
ing from static pixel-wise classification to a dynamic
decision-making paradigm.
• We develop a hybrid prompting strategy for expert-
curated trajectory generation, followed by a two-stage
training pipeline optimized via a clinical-fidelity pro-
cess reward design.
• We conduct comprehensive experiments across 6 med-
ical modalities that demonstrate superior segmentation
performance and robust tool-agnostic generalization.
2. Related Works
2.1. Medical Image Segmentation
Traditional medical image segmentation methods like UNet
```
(Ronneberger et al., 2015), nn-UNet (Isensee et al., 2021),
```
```
and other representative architectures (Li et al., 2025a; Zhou
```
```
et al., 2018; Chen et al., 2021; Fan et al., 2020) are inher-
```
ently task-specific, which require dedicated training for each
individual segmentation task, thus exhibiting limited gener-
```
alization across diverse medical imaging scenarios (Butoi
```
```
et al., 2023; Ye et al., 2023). The advent of the SAM (Kir-
```
```
illov et al., 2023; Ravi et al., 2024) has sparked renewed
```
hope for achieving generalizable medical image segmenta-
```
tion. Consequently, numerous SAM-derived models (Ma
```
```
et al., 2024; Wei et al., 2024b; Konwer et al., 2025; Zhang
```
```
et al., 2025c; Ma et al., 2025b; Wu et al., 2025a; Wang et al.,
```
```
2024; Zhao et al., 2025) have been fine-tuned on large-scale
```
medical image datasets. However, these interactive seg-
mentation models typically rely on explicit human-provided
```
prompts (e.g., points, boxes, or masks) to guide the seg-
```
mentation process, limiting their autonomy. Parallel to this,
the advancement of MLLMs has enabled text-guided medi-
```
cal image segmentation (Huang et al., 2025b; Wang et al.,
```
2
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
```
2025b; Ning et al., 2025), such as Citrus-V (Wang et al.,
```
```
2025a) and UniBiomed (Wu et al., 2025b). Despite their
```
flexibility in leveraging textual instructions, these MLLM-
driven methods are constrained by the inherent limitations
of MLLMs in capturing fine-grained pixel-level details, and
they lack the iterative refinement capability that character-
izes SAM-based interactive segmentation frameworks. In
contrast, our proposed method addresses these gaps by mim-
icking the annotation trajectories of human experts through
the integration of interactive segmentation tools, enabling
both autonomous guidance and iterative optimization.
2.2. Reinforcement Learning in MLLM
```
Driven by the success of DeepSeek-R1 (Shao et al., 2024),
```
```
reinforcement learning (RL) has fundamentally reshaped
```
```
the training paradigms for large language models (Schulman
```
```
et al., 2017; Yu et al., 2025; Zheng et al., 2025a; Zhang et al.,
```
```
2020). Recent studies (Peng et al., 2025; Shen et al., 2025a;
```
```
Feng et al., 2025; Liu et al., 2025d;b;d) have demonstrated
```
that simple yet verifiable rewards can effectively extend a
model’s reasoning capabilities from text-only tasks to multi-
modal scenarios, a transition that has also proven effective
```
in specialized medical tasks (Wang et al., 2025c; Lai et al.,
```
```
2025b; Pan et al., 2025; Zhang et al., 2025b; Chen et al.,
```
```
2025; Xu et al., 2025a). For fine-grained understanding
```
```
tasks, milestone developments like OpenAI-o3 (OpenAI,
```
```
2025) have introduced the concept of “thinking with im-
```
```
ages”, leveraging iterative zoom-in mechanisms (Zheng
```
```
et al., 2025b; Lai et al., 2025a; Shen et al., 2025b; Su et al.,
```
```
2025a; Zhang et al., 2025d) or external tool invocation (Su
```
```
et al., 2025b; Huang et al., 2025a; You & Wu, 2025; Wu
```
```
et al., 2025c; Zhang et al., 2025a) to actively refine visual
```
perception. In the medical domain, while similar efforts
```
(Huang et al., 2025c; Yan et al., 2025; Li et al., 2024; Wang
```
```
et al., 2025d) have attempted to integrate tool-use, exist-
```
```
ing medical MLLMs (Li et al., 2024; Jiang et al., 2026;
```
```
2025b) predominantly operate within a single-turn interac-
```
tion paradigm or rely on monotonic clicking strategies that
lack strategic diversity. Consequently, their potential to func-
tion as autonomous agents capable of dynamic, multi-step
action planning remains underdeveloped.
3. Methods
3.1. Expert-Curated Trajectory Generation
Given an image I and a segmentation target prompt P ,
medical experts commonly employ interactive segmentation
```
tools: based on the original image I and the real-time up-
```
dated mask M , they iteratively provide positive click points
```
(marking regions confirmed as the target) and negative click
```
```
points (excluding non-target regions). This iterative interac-
```
tion facilitates the efficient generation of high-quality masks
that meet clinical annotation standards. We formalize this
procedure as a multi-step decision-making process. Given
the prompt P and input image I, the policy model iteratively
generates an action at. This action interacts with the envi-
ronment by invoking the image segmentation tool, resulting
in a new observation ot. This observation is appended to
the interaction history and fed back to the policy model
```
πθ (at|st, I, P ). The components are detailed as follows:
```
Action at: To simulate expert behavior, we define the action
space to encompass bounding box operations, point-wise
clicks, and a stop signal. The box operation is represented as
```
at = [x1, y1, x2, y2] to provide a global spatial prior, while
```
a click operation consists of a coordinate at = [x1, y1]
```
paired with an attribute α ∈ {+1, −1} to denote positive or
```
```
negative refinement; finally, the stop operation is triggered
```
to terminate the sequence once the segmentation reaches the
desired fidelity.
Observation ot: In agentic RL, the observation ot is ob-
tained based on the parameters indicated by the correspond-
ing action at. Concretely, it is the updated mask Mt pre-
dicted by the interactive segmentation network Fseg . Prior
to the execution of a stop action, the interactive model
maintains a record of all previous actions, with the up-
```
dated mask computed as Mt = Fseg (I, ot−1, at), where
```
Mt directly serves as ot. These observation tokens are ap-
pended to the ongoing rollout sequence and fed back into
the model as input for the subsequent step. When the action
```
at = stop action, no tool invocation is performed; instead,
```
ot directly inherits the result from the previous iteration
ot−1 and is output as the final predicted mask.
State st: At each step t, the state st is defined as:
```
st = {(a0, o0), (a1, o1), . . . , (at−1, ot−1)} (1)
```
```
Given the state st, the action at ∼ πθ (a | st) is sampled
```
from the MLLM policy πθ , serving as the next input token.
This long sequence continues to interleave until either the
stop action is generated or the maximum number of tool
calls is reached. The objective of our framework is to de-
velop an MLLM capable of executing a segmentation policy
```
πθ (at|st, I, P ) that emulates the decision-making process
```
of medical experts. To facilitate this, we construct a trajec-
```
tory dataset Dtraj = {(I, Mtarget, P, st)}, derived from an
```
```
initial dataset Dseg = {(I, Mtarget, P )}.
```
```
Hybrid Prompting Strategy. While existing methods (Zhu
```
```
et al., 2025; Jiang et al., 2026) rely on a rigid point-wise
```
simulation driven by pixel-level discrepancies, they typi-
cally restrict the action space to sequential clicks. Such a
limitation fails to capture the multi-modal nature of human
workflows, where practitioners often initiate segmentation
by defining an ROI. To bridge this gap, we propose a hy-
```
brid prompting strategy (Fig. 2) that offers greater spatial
```
flexibility. This strategy incorporates both Box-to-Point
and Sequential-Click paradigms to better mimic clinical
3
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
```
Question: The target to be segmented is polyp.
```
```
Find the polyp and locate by bbox{“name”:“add_bbox”, “arguments”:
```
```
{"bbox_2d": [195, 43, 587, 491]}}
```
```
Need Refinement. {"name":"add_point","arguments": {"point_2d": [198,401],
```
```
"point_type": "positive"}}
```
```
Still need Refinement. {“name”:“add_point”, “arguments”: {"point_2d":
```
```
[211,480], "point_type": "positive"}}
```
```
Good enough to stop. {"name":"stop_action", "arguments": {}}
```
Trajectory Generation
Image-Label Pairs
Click Simulation
Stage 1 SFT Cold-Start Stage2 RL
```
Input a0 o0(I, P) a1 o1
```
oias aiOutputMask M
Policy
model
..
.
→ →
→→..
.
→
Rollout 1
→
<tool_call> IoU/Dice
Progress reward
Max IoU > Final IoU
Turn n+1 IoU > Turn n IoU
Tool-using cost
Format reward Quality reward
→Resample Stop
IoU Filter Policy Optimization
ai
N
T > Tretry
ai+1
Box/PointInit a
0→
→Y
→Quality Check∆IoU > τ ?
Initial Refinement Finish
→→
Trajectory
Point
Box
Stop
```
{[x, y], Positive/Negative}
```
```
{[x1,y1, x2, y2]}
```
Multi-turn Interaction
Rollout 2
...
Rollout n
🔥
Training
Reward Design
→
Seg Tools
ai
Seg
Modelai+1
as
```
{oi, ai}
```
Figure 2. Overview of MedSAM-Agent. We develop a hybrid prompting strategy for expert-curated trajectory generation that transforms
image-label pairs into high-quality interaction sequences via simulated clicks and IoU-based filtering. Then these trajectories support
a two-stage training pipeline, stage-1 SFT cold-start for initial capability and stage-2 RL optimized by a fine-grained reward design.
MedSAM-Agent can autonomously select between box and point tools and execute the “stop” action once the refinement is complete.
expertise. In the Box-to-Point workflow, the trajectory is ini-
tialized with a bounding box action generated by extracting
the axis-aligned rectangle of Mtarget with a controllable ran-
dom jitter to simulate human imprecision. For subsequent
refinement, the agent identifies the most significant False
```
Negative (FN) and False Positive (FP) components. By
```
applying a distance transform to these error regions, correc-
tive clicks are sampled at the centroids of the largest error
clusters, ensuring each action addresses the most salient
morphological defects.
To guarantee the quality and efficiency of the synthesized
trajectories, we implement a progress-constrained sampling
mechanism. We contend that an expert-level trajectory must
```
consist of informative and effective actions; consequently,
```
we enforce a constraint where each simulated action must
```
yield an incremental IoU gain (∆IoU ) exceeding a prede-
```
fined threshold τ . To prevent the inclusion of sub-optimal
or redundant interactions, a retry mechanism is introduced:
if a candidate action fails to satisfy the progress threshold,
the simulator performs iterative resampling up to N trials
to identify a more constructive interaction.Following the
generation of all sequences, a global IoU filter is applied
to both paradigms to prune trajectories that fail to reach a
target performance threshold. This validation process ef-
fectively filters out stochastic noise and ensures that Dtraj
is composed of high-quality, monotonically improving se-
quences, thereby providing a robust supervisory signal for
policy optimization. By integrating these components, we
successfully construct the expert-curated trajectories Dtraj.
The details of the trajectory generation algorithm are shown
in the Appendix.
3.2. Two-stage Training Pipeline
Building upon these trajectories, we employ a two-stage
```
training pipeline: Supervised Fine-Tuning (SFT) serves as
```
the initial cold-start, followed by Reinforcement Learning
```
with Verifiable Rewards (RLVR) in the second stage to fur-
```
ther refine the agent’s decision-making policy. The overall
training pipeline is shown in Fig. 2.
3.2.1. SFT COLD-START
To bridge the gap between natural language reasoning and
tool interaction, we format the trajectory data into a struc-
tured tool-calling schema. Each expert action at is con-
verted into a specialized token sequence enclosed within
<tool call> tags, resulting in a SFT dataset DSFT com-
prising 449K samples. The system prompt P encompasses
a comprehensive description of the task requirements and
action space, alongside a semantic definition of the target
object. This design ensures the model maintains a global
objective while grounding its actions in the specific visual
and textual context of the segmentation target. The details
of the prompt design are shown in the Appendix.
As each expert-curated action is tokenized into a discrete
sequence, the model is trained to predict the next token
in an auto-regressive manner. Formally, the optimization
objective is defined by the following negative log-likelihood
```
loss:
```
```
LSFT(θ) = −E(I,P,y)∼Dsft
```


|y|X
```
t=1
```
```
log πθ (yt|I, P, y<t)
```

```
 (2)
```
4
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
where y denotes the target token sequence representing the
expert trajectory sT , yt is the t-th token of the sequence,
and y<t represents the preceding context including the im-
age I, the text prompt P , and previously generated tokens.
To provide the model with a precise perception of the cur-
rent segmentation state, the mask generated at each step
is overlaid on the original image I as a dedicated visual
prompt, which is then re-encoded by the vision encoder to
inform the subsequent action. This iterative visual feedback
loop enables the agent to dynamically observe the refined
boundaries and localize remaining errors throughout the
trajectory.
This cold-start phase is foundational for the subsequent re-
```
inforcement learning stage; by minimizing the imitation
```
loss, the model aligns its initial policy distribution with the
decision-making heuristics of medical experts, effectively
acquiring fundamental visual grounding capabilities. This
strategic initialization ensures that the agent enters the rein-
forcement learning phase with a pre-established competency
in autonomous tool invocation and spatial reasoning.
3.2.2. CLINICAL-FIDELITY PROCESS REWARD DESIGN
```
Unlike prior approaches (Liu et al., 2025b; Huang et al.,
```
```
2025a; You & Wu, 2025) that predominantly rely on
```
outcome-based rewards for single-pass predictions, we pro-
pose a multi-dimensional reward framework. This design
meticulously balances final segmentation fidelity with the
strategic efficiency of the interactive process, fostering a
policy that is both accurate and resource-conscious.
```
Format Reward (Rf mt): The format reward Rf mt is en-
```
gineered to enforce adherence to the interactive protocol
through a partial-credit mechanism. It rewards two funda-
```
mental behaviors: (1) the active deployment of interaction
```
```
primitives (e.g., add bbox or add point) and (2) the execu-
```
tion of a definitive stop action to terminate the sequence.
By assigning 0.5 points to each criterion, we prevent the
model from collapsing into infinite loops, ensuring the agent
learns the logical boundaries of task completion.
```
Quality Reward (Rqual): The core objective is to maxi-
```
mize the final intersection between the predicted mask and
the ground truth. We define the quality reward Rqual as a
weighted ensemble of the IoU and the Dice Coefficient:
```
Rqual = wiou · IoUf inal + wdice · Dicef inal (3)
```
wiou and wdice are weighted hyper-parameters. While IoU
provides a standard geometric consistency measure, the
inclusion of the Dice coefficient is critical for medical con-
texts, as it offers higher sensitivity to small-scale lesions
and anatomical structures, thereby mitigating the penalty
variance in class-imbalanced scenarios.
To optimize the agent’s decision-making trajectory, we in-
troduce a process reward comprised of three distinct com-
ponents designed to foster intelligent interaction logic:
```
Progressive Improvement Bonus (Rimp): This compo-
```
nent incentivizes the agent to achieve monotonic quality
gains across iterations. By rewarding the cumulative posi-
tive deltas in IoU, we discourage redundant or ineffective
interactions that fail to contribute to mask refinement.
```
Rimp =
```
N −1X
```
t=1
```
```
max(0, IoUt − IoUt−1) (4)
```
```
Overshoot Penalty (Rover ): This penalty is designed to
```
encourage efficient termination and prevent redundant inter-
actions. Since the model lacks access to ground-truth IoU
during inference, it must internalize the ability to decide
when to conclude an interaction based solely on visual cues.
By penalizing any quality degradation following the peak
```
IoU ( IoUmax −IoUf inal), during training, Rover effectively
```
disciplines the agent to recognize the point of diminishing
returns. This mechanism ensures the model learns an opti-
mal stopping policy, minimizing unnecessary iterations and
maximizing inference efficiency in real-world scenarios.
```
Tool-cost Penalty (Rcost): Recognizing the computational
```
cost of each interaction, we apply a linear penalty propor-
tional to the sequence length. This encourages the policy to
pursue the shortest path to high-fidelity segmentation, prior-
itizing efficiency without compromising terminal accuracy.
The terminal reward Rtotal for each output Oi is a compos-
ite metric that accounts for structural adherence, segmenta-
tion accuracy, and procedural efficiency:
```
Rtotal = w1 · Rf mt + w2 · clip(Rqual
```
```
+λ1Rimp − λ2Rover − λ3Rcost, 0, 1) (5)
```
```
where w and λ are weighted hyper-parameters, clip(·, a, b)
```
limits the input to the range [a, b]. To optimize the inter-
active policy, we employ Group Relative Policy Optimiza-
```
tion (GRPO (Shao et al., 2024)). The objective function
```
```
JGRP O (θ) is defined by sampling a group of G rollouts
```
```
{O1, O2, . . . , OG} for each query q:
```
```
JGRP O (θ) =Eq∼D,{Oi}Gi=1∼πθold
```
 1
G
GX
```
i=1
```
min

```
πθ (Oi|q)
```
```
πθold (Oi|q) Ai,clip
```
 π
```
θ (Oi|q)
```
```
πθold (Oi|q) , 1 − ϵ, 1 + ϵ
```

Ai

```
(6)
```
where the advantage Ai is computed by normalizing the
multi-component reward Rtotal within each group to cap-
ture the relative quality of the rollouts Oi:
```
Ai = Rtotal,i − mean(Rtotal,1, . . . , Rtotal,G)std(R
```
```
total,1, . . . , Rtotal,G)
```
```
(7)
```
5
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
Table 1. Quantitative comparison across six medical imaging modalities. The results report the mean Dice and IoU scores for each
specific modality and the overall average. Point and Box indicate the interactive prompts derived from ground-truth center points and
bounding boxes, used to evaluate single-round segmentation performance. Bold and underlined values represent the best and second-best
performance among non-interactive methods, respectively.
Method CT MRI X-Ray Ultrasound Fundus Endoscopy Avg
Dice IoU Dice IoU Dice IoU Dice IoU Dice IoU Dice IoU Dice IoU
SAM based methods
```
SAM2-Point (Ravi et al., 2024) 0.610 0.531 0.488 0.412 0.546 0.422 0.528 0.408 0.362 0.265 0.659 0.597 0.532 0.439
```
```
SAM2-Box (Ravi et al., 2024) 0.837 0.754 0.838 0.748 0.827 0.733 0.871 0.779 0.717 0.574 0.922 0.872 0.835 0.743
```
```
MedSAM-Point (Ma et al., 2025b) 0.747 0.640 0.667 0.584 0.843 0.776 0.591 0.467 0.414 0.346 0.822 0.736 0.681 0.592
```
```
MedSAM-Box (Ma et al., 2025b) 0.847 0.759 0.854 0.768 0.897 0.836 0.890 0.806 0.846 0.743 0.925 0.872 0.876 0.797
```
```
IMISNet-Point (Cheng et al., 2025) 0.674 0.580 0.678 0.556 0.578 0.449 0.489 0.371 0.677 0.528 0.781 0.681 0.646 0.527
```
```
IMISNet-Box (Cheng et al., 2025) 0.805 0.727 0.849 0.758 0.706 0.628 0.569 0.483 0.841 0.733 0.905 0.846 0.779 0.696
```
MLLM based methods
```
LISA (Lai et al., 2024) 0.088 0.051 0.110 0.063 0.322 0.230 0.385 0.280 0.036 0.019 0.282 0.222 0.204 0.144
```
```
GLAMM (Rasheed et al., 2024) 0.088 0.053 0.106 0.061 0.285 0.194 0.402 0.294 0.027 0.014 0.245 0.179 0.192 0.132
```
```
HyperSeg (Wei et al., 2024a) 0.098 0.069 0.153 0.101 0.336 0.231 0.434 0.336 0.023 0.012 0.396 0.342 0.240 0.182
```
```
Seg-R1 (You & Wu, 2025) 0.099 0.064 0.126 0.075 0.417 0.318 0.490 0.382 0.455 0.325 0.533 0.471 0.353 0.272
```
```
MedPLIB (Huang et al., 2025b) 0.052 0.038 0.177 0.136 0.115 0.075 0.088 0.056 0.298 0.201 0.132 0.097 0.144 0.101
```
```
UniBiomed (Wu et al., 2025b) 0.724 0.634 0.807 0.716 0.817 0.748 0.736 0.618 0.794 0.668 0.778 0.704 0.776 0.681
```
```
Citrus-V(Wang et al., 2025a) 0.363 0.302 0.326 0.258 0.624 0.536 0.134 0.094 0.782 0.655 0.666 0.616 0.482 0.410
```
```
Qwen3-VL+MedSAM2 (RL only) 0.401 0.346 0.647 0.559 0.834 0.779 0.718 0.600 0.775 0.644 0.695 0.631 0.678 0.593
```
Ours-IMISNet 0.732 0.654 0.783 0.685 0.798 0.723 0.658 0.547 0.803 0.678 0.803 0.736 0.763 0.670
Ours-MedSAM2 0.720 0.633 0.793 0.701 0.833 0.775 0.793 0.685 0.813 0.692 0.811 0.744 0.794 0.705
By leveraging this relative advantage formulation, we ensure
that the model learns to prioritize higher-reward trajectories.
Through carefully designed RL training, the agent becomes
capable of interpreting complex target descriptions and per-
forming pixel-level reasoning via iterative tool invocations.
Moreover, we specifically curate 9K high-quality samples
for the RL training data to effectively guide this learning
process and ensure strong policy convergence during the
RL rollout. This selection focuses on challenging instances
that necessitated 3–5 interaction rounds under both Box-to-
Point and Sequential-Click paradigms, thereby exposing the
model to complex decision-making scenarios and maximiz-
ing the efficiency of the optimization strategy.
4. Experiments
4.1. Experiments Setting
Datasets. We utilize a comprehensive collection of 21 open-
```
source datasets spanning 6 modalities, including CT (Land-
```
```
man et al., 2015; Ji et al., 2022; Ma et al., 2022; Heller et al.,
```
```
2023; Armato III et al., 2011), MRI (Bernard et al., 2018; Ji
```
```
et al., 2022; Buda et al., 2019), X-Ray (Jaeger et al., 2014;
```
```
Chowdhury et al., 2020; Khaled et al., 2021), Ultrasound
```
```
(Vitale et al., 2020; Jieyun & ZhanHong, 2024), Fundus
```
```
(Orlando et al., 2020), and Endoscopy (Ngoc Lan et al.,
```
```
2021; Ali et al., 2023). All data formats are standardized
```
```
following previous methods (Zhao et al., 2025; Wu et al.,
```
```
2025b). Details are available in the Appendix.
```
Implementation Details. In our experiments, the policy
```
model is based on the Qwen3-VL-8B (Bai et al., 2025),
```
and the interactive segmentation models used in our work
```
include general-purpose model SAM2.1-Base (Ravi et al.,
```
```
2024), and medical domain-specific models MedSAM2 (Ma
```
```
et al., 2025b) and IMISNet (Cheng et al., 2025). The SFT
```
```
stage is implemented using Llama-Factory (Zheng et al.,
```
```
2024b) with a learning rate of 1e-5 and the batch size is 64.
```
```
The RL stage is implemented by the Verl framework (Sheng
```
```
et al., 2024) with a learning rate of 1e-5, batch size of 8 and
```
sampling number of 8, the maximum interaction turn is 5.
All experiments are conducted on 8× NVIDIA H20 GPUs.
4.2. Comparison Experiment
Baselines and Protocols. In this section, we conduct a com-
prehensive evaluation of our MedSAM-Agent against exist-
```
ing state-of-the-art (SOTA) methods, including the interac-
```
```
tive segmentations like general-purpose model SAM2 (Ravi
```
```
et al., 2024) and domain-specific models such as Med-
```
```
SAM2 (Ma et al., 2025b) and IMISNet (Cheng et al., 2025).
```
For these baselines, we report results using both Point and
Box prompts. “Point” results are generated using the cen-
ter of the ground-truth mask, while “Box” results utilize
corresponding bounding boxes. The Box prompt perfor-
mance is generally regarded as the empirical upper bound
for single-turn interactive segmentation. Furthermore, we
6
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
Figure 3. Analysis of multi-turn interaction. The orange and
purple lines represent the performance of the static Single-Turn
Box and Single-Turn Point prompts, respectively, where inputs are
derived from ground-truth bounding boxes and centroids. The blue
line plots the Mean IoU across successive interaction turns. The
bar charts illustrate the distribution of sample outcomes at each
turn, where green, red, and grey segments denote the proportion
of samples exhibiting improved, declined, or unchanged IoU. The
```
segmentation tool is IMISNet (Cheng et al., 2025).
```
benchmark against SOTA MLLM-based methods, includ-
```
ing four general-domain models (LISA (Lai et al., 2024),
```
```
GLAMM (Rasheed et al., 2024), HyperSeg (Wei et al.,
```
```
2024a), and Seg-R1 (You & Wu, 2025)) and three special-
```
```
ized medical MLLMs (MedPLIB (Huang et al., 2025b),
```
```
UniBioMed (Wu et al., 2025b), and Citrus-V (Wang et al.,
```
```
2025a)).
```
Segmentation Performance. Table 1 shows the quantitative
comparison across 6 medical imaging modalities. We ob-
```
serve that general-purpose MLLMs, such as LISA (Lai et al.,
```
```
2024), exhibit poor performance on these medical datasets.
```
This suggests that the <seg> token-based paradigm suffers
from limited generalization and faces significant hurdles
when transferring to the specialized medical domain with-
out extensive in-domain fine-tuning. In contrast, although
```
Seg-R1 (You & Wu, 2025) was not specifically fine-tuned
```
on medical domain data, it significantly outperforms other
general-purpose MLLMs. This suggests that RL-based op-
timization inherently equips models with superior struc-
tural reasoning and out-of-distribution generalization for
segmentation tasks, even when navigating unfamiliar med-
ical modalities. Furthermore, specialized medical models
such as UniBioMed and Citrus-V show marked improve-
ments over general-purpose baselines, underscoring the vital
importance of domain-specific medical knowledge. Build-
Table 2. Performance comparison of various policy models
```
and segmentation frameworks. (Tr/Te denotes the segmentation
```
```
models used during training and testing, respectively; ‘-’ indicates
```
zero-shot evaluation.
```
Policy Model Seg Model (Tr/Te) BTCV PolypGen
```
Dice IoU Dice IoU
GPT-4o - / MedSAM2 0.122 0.104 0.549 0.481
Qwen2.5-VL-7B - / MedSAM2 0.117 0.098 0.389 0.331
Qwen3-VL-8B - / MedSAM2 0.130 0.113 0.669 0.605
Qwen3-VL-8B - / IMIS 0.101 0.080 0.539 0.466
Qwen3-VL-8B IMIS / IMIS 0.780 0.702 0.793 0.719
Qwen3-VL-8B MedSAM2 / MedSAM2 0.773 0.690 0.809 0.735
Qwen3-VL-8B IMIS / SAM2 0.764 0.680 0.805 0.740
Qwen3-VL-8B IMIS / MedSAM2 0.769 0.687 0.799 0.726
Qwen3-VL-8B MedSAM2 / IMIS 0.761 0.682 0.798 0.720
Qwen3-VL-8B MedSAM2 / SAM2 0.763 0.677 0.808 0.739
ing upon this, MedSAM-Agent further extends these gains
by internalizing multi-turn interaction logic, consistently
achieving high mask fidelity across all evaluated datasets.
Details of the segmentation performance of 21 datasets are
shown in the Appendix.
Moreover, we analyze the transformative power of the multi-
turn strategic interaction of MedSAM-Agent. To demon-
strate the efficacy of autonomous interaction, we compare
our agent against static baseline paradigms, specifically
Single-Turn Point and Single-Turn Box prompts where in-
puts are directly derived from ground-truth centroids and
bounding boxes. As illustrated in Fig. 3, which details
```
the performance trajectories on the FH-PS-AOP (Jieyun
```
```
& ZhanHong, 2024) and Radiography (Chowdhury et al.,
```
```
2020) datasets, MedSAM-Agent outperforms these ideal
```
single-turn baselines. Notably, the Final Mean IoU achieved
by our agent exceeds the theoretical upper bound of static
prompts across all evaluated modalities, proving that the
agent has internalized a sophisticated refinement logic rather
than simple tool invocation. The stacked bar charts further
quantify this effectiveness: the predominance of green seg-
ments signifies that the vast majority of autonomous inter-
actions result in substantial IoU gains. Furthermore, the
diminishing sample counts across successive turns confirm
the agent’s ability to adaptively select the optimal inter-
action depth, effectively balancing segmentation precision
with procedural efficiency.
Zero-shot Tool Agnosticism. To investigate the model’s
ability to generalize across various interactive segmenta-
tion backends, we conducted a cross-tool evaluation by
separately synthesizing training trajectories using two dis-
tinct models, MedSAM2 and IMISNet, and subsequently
testing the learned policies against alternative engines in-
cluding SAM2. Our experiments reveal that while trajec-
tories curated via MedSAM2 generally result in superior
7
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
agent performance due to its higher fidelity in mask bound-
ary delineations, the resulting agent exhibits exceptional
zero-shot generalization regardless of the training back-
end. As summarized in Table 2, an agent trained on one
specific tool can be seamlessly interfaced with alternative
backends at inference time with negligible performance
degradation. This tool-agnostic capability suggests that
the RL-based policy has successfully decoupled high-level
strategic decision-making involving optimal point place-
ment and autonomous stopping criteria from the low-level
mask generation mechanics of specific tools. Consequently,
MedSAM-Agent successfully internalizes a universal in-
teraction logic, demonstrating its robust transferability and
potential for integration into diverse clinical workflows in-
dependent of the underlying segmentation engine.
4.3. Ablation Study
Effectiveness of Action Strategies. We first investigate
the impact of different interaction strategies on segmenta-
tion performance. As illustrated in Table 3, the Hybrid
strategy clearly outperforms any single-strategy approach.
Compared to a purely Sequential-Click strategy, the use
of the box-to-point strategy alone significantly improves
the IoU from 0.623 to 0.649 while reducing the average
interaction turns from 2.94 to 2.06. This demonstrates that
the box strategy provides a superior spatial constraint that
effectively defines target boundaries with fewer interactions.
Most importantly, the Hybrid strategy achieves the highest
performance among the SFT variants, reaching an IoU of
0.686. This success stems from the fact that the hybrid
approach aligns with human intuition: it leverages the box
to establish a reliable global context and uses subsequent
points for precise local refinement. These results prove that
combining box and point strategies is the most effective
way to navigate complex medical lesions, as it minimizes
redundant actions while maximizing segmentation fidelity.
Analysis of Training Strategies. We evaluate three train-
ing paradigms: direct RL from scratch, SFT only, and our
```
combined SFT+RL approach (Table 3). Direct RL exhibits
```
a clear performance dichotomy across different modalities.
In scenarios with prominent lesions and relatively simple
backgrounds, such as X-ray and Endoscopy, direct RL can
achieve competitive results as the agent easily identifies the
target regions. However, it fails to achieve precise localiza-
tion in more challenging modalities like CT and MRI, where
the vast search space and low tissue contrast pose signifi-
```
cant obstacles (Table. 1). This discrepancy highlights that
```
while RL is adept at strategy optimization, it struggles with
the initial grounding problem in complex medical contexts.
In contrast, our SFT cold-start phase establishes essential
anatomical grounding and domain knowledge. The subse-
quent RL stage then enables the model to generalize across
tools and optimize for procedural efficiency. This two-stage
Table 3. Ablation study. Metrics are average Dice and IoU, and
```
the segmentation tool is MedSAM2 (Ma et al., 2025b).
```
Setting Dice IoU nturn
Qwen3-VL-8B base 0.263 0.178 4.34
- Cold-start SFT only (Hybrid) 0.769 0.686 2.35
- only Sequential-Click 0.719 0.623 2.94
- only Box-to-Point 0.745 0.649 2.06
- RL only 0.678 0.593 2.23
- Cold-start SFT + RL (Full) 0.794 0.705 2.11
- w/o Rimp 0.772 0.688 2.08
- w/o Rover 0.785 0.693 2.24
- w/o Rcost 0.788 0.696 2.26
pipeline ensures the agent possesses both the anatomical
awareness required for medical precision and the strategic
flexibility for efficient, multi-step interaction.
Impact of Reward Design. To evaluate the contribution
of each component in our proposed reward framework, we
conduct an ablation study on the process-aware rewards as
shown in Table 3. Our results indicate that while the base-
line model achieves basic segmentation, it often suffers from
inefficient interaction trajectories and redundant operations.
The inclusion of the Rimp is critical for maintaining mask
```
fidelity throughout the interaction; its removal results in a
```
decline in Dice score from 0.794 to 0.772 and a drop in
IoU to 0.688. The Rover serves as a vital regularizer for the
agent to determine the optimal timing for task completion.
When Rover is excluded, the agent fails to terminate the
process at optimal states, leading to an increase in average
```
interaction turns (nturn) from 2.11 to 2.24. Similarly, the
```
absence of the Rcost confirms that our process-level super-
vision effectively discourages redundant strategies. Overall,
the two-stage pipeline achieves the superior balance of preci-
sion and interaction turns, demonstrating that process-aware
rewards successfully align the agent’s behavior with both
clinical accuracy and operational efficiency.
5. Conclusion
In this paper, we propose MedSAM-Agent, a novel frame-
work that shifts the medical image segmentation paradigm
from static classification to an autonomous, multi-step
decision-making process. By addressing the limitations
of current interactive models that depend on human guid-
ance and automated MLLM approaches that lack iterative
refinement, our method successfully bridges the gap be-
tween high-level reasoning and precise tool interaction.
Through a hybrid prompting strategy and a two-stage train-
ing pipeline optimized by fine-grained process rewards, we
enable the model to function as a self-refinement agent ca-
pable of achieving expert-level segmentation across diverse
8
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
imaging modalities. Experimental results underscore the
framework’s superior performance and robust generaliza-
tion, demonstrating its potential to reduce the workload of
clinical professionals.
References
Ali, S., Jha, D., Ghatwary, N., Realdon, S., Cannizzaro, R.,
Salem, O. E., Lamarque, D., Daul, C., Riegler, M. A.,
Anonsen, K. V., et al. A multi-centre polyp detection
and segmentation dataset for generalisability assessment.
```
Scientific Data, 10(1):75, 2023.
```
Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray,
M. F., Meyer, C. R., Reeves, A. P., Zhao, B., Aberle, D. R.,
Henschke, C. I., Hoffman, E. A., et al. The lung image
```
database consortium (lidc) and image database resource
```
```
initiative (idri): a completed reference database of lung
```
```
nodules on ct scans. Medical physics, 38(2):915–931,
```
2011.
Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang,
K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y.,
Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu,
Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang,
Z., Xu, H., and Lin, J. Qwen2.5-vl technical report. arXiv
preprint arXiv:2502.13923, 2025.
Bernard, O., Lalande, A., Zotti, C., Cervenansky, F., Yang,
X., Heng, P.-A., Cetin, I., Lekadir, K., Camara, O.,
Ballester, M. A. G., et al. Deep learning techniques for
automatic mri cardiac multi-structures segmentation and
```
diagnosis: is the problem solved? IEEE Transactions on
```
```
Medical Imaging, 37(11):2514–2525, 2018.
```
Buda, M., Saha, A., and Mazurowski, M. A. Association
of genomic subtypes of lower-grade gliomas with shape
features automatically extracted by a deep learning algo-
rithm. Computers in Biology and Medicine, 109:218–225,
2019.
Butoi, V. I., Ortiz, J. J. G., Ma, T., Sabuncu, M. R., Guttag,
J., and Dalca, A. V. Universeg: Universal medical image
segmentation. In ICCV, pp. 21438–21451, 2023.
Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu,
L., Yuille, A. L., and Zhou, Y. Transunet: Transformers
make strong encoders for medical image segmentation.
arXiv preprint arXiv:2102.04306, 2021.
Chen, J., Gui, C., Ouyang, R., Gao, A., Chen, S., Chen,
G. H., Wang, X., Zhang, R., Cai, Z., Ji, K., Yu, G., Wan,
X., and Wang, B. Huatuogpt-vision, towards injecting
medical visual knowledge into multimodal llms at scale,
2024a.
Chen, K., Rui, S., Jiang, Y., Wu, J., Zheng, Q., Song, C.,
Wang, X., Zhou, M., and Liu, M. Think twice to see
```
more: Iterative visual reasoning in medical vlms. arXiv
```
preprint arXiv:2510.10052, 2025.
Chen, W., Shen, L., Lin, J., Luo, J., Li, X., and Yuan, Y.
Fine-grained image-text alignment in medical imaging
enables explainable cyclic image-report generation. In
Proceedings of the 62nd Annual Meeting of the Associ-
```
ation for Computational Linguistics (Volume 1: Long
```
```
Papers), pp. 9494–9509, 2024b.
```
Cheng, J., Fu, B., Ye, J., Wang, G., Li, T., Wang, H., Li, R.,
Yao, H., Cheng, J., Li, J., et al. Interactive medical image
```
segmentation: A benchmark dataset and baseline. In Pro-
```
ceedings of the Computer Vision and Pattern Recognition
Conference, pp. 20841–20851, 2025.
Chowdhury, M. E., Rahman, T., Khandakar, A., Mazhar,
R., Kadir, M. A., Mahbub, Z. B., Islam, K. R., Khan,
M. S., Iqbal, A., Al Emadi, N., et al. Can ai help in
screening viral and covid-19 pneumonia? IEEE Access,
8:132665–132676, 2020.
Fan, D.-P., Ji, G.-P., Zhou, T., Chen, G., Fu, H., Shen, J.,
and Shao, L. Pranet: Parallel reverse attention network
for polyp segmentation, 2020. URL https://arxiv.
org/abs/2006.11392.
Fathi, N., Kumar, A., and Arbel, T. Aura: A multi-modal
medical agent for understanding, reasoning and anno-
tation. In International Workshop on Agentic AI for
Medicine, pp. 105–114. Springer, 2025.
Feng, K., Gong, K., Li, B., Guo, Z., Wang, Y., Peng, T.,
Wu, J., Zhang, X., Wang, B., and Yue, X. Video-r1:
Reinforcing video reasoning in mllms. arXiv preprint
```
arXiv:2503.21776, 2025.
```
Gong, S., Zhong, Y., Ma, W., Li, J., Wang, Z., Zhang,
J., Heng, P.-A., and Dou, Q. 3dsam-adapter: Holistic
adaptation of sam from 2d to 3d for promptable tumor
segmentation. Medical Image Analysis, 98:103324, 2024.
He, X., Zhang, Y., Mou, L., Xing, E., and Xie, P. Pathvqa:
30000+ questions for medical visual question answering.
arXiv preprint arXiv:2003.10286, 2020.
Heller, N., Isensee, F., Trofimova, D., Tejpaul, R., Zhao, Z.,
Chen, H., Wang, L., Golts, A., Khapun, D., Shats, D., et al.
The kits21 challenge: Automatic segmentation of kidneys,
renal tumors, and renal cysts in corticomedullary-phase
ct. arXiv preprint arXiv:2307.01984, 2023.
Hu, Y., Li, T., Lu, Q., Shao, W., He, J., Qiao, Y., and
Luo, P. Omnimedvqa: A new large-scale comprehensive
evaluation benchmark for medical lvlm. In Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 22170–22183, 2024.
9
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
Huang, J., Xu, Z., Zhou, J., Liu, T., Xiao, Y., Ou, M., Ji, B.,
Li, X., and Yuan, K. Sam-r1: Leveraging sam for reward
feedback in multimodal segmentation via reinforcement
learning. arXiv preprint arXiv:2505.22596, 2025a.
Huang, X., Shen, L., Liu, J., Shang, F., Li, H., Huang, H.,
and Yang, Y. Towards a multimodal large language model
with pixel-level insight for biomedicine. In Proceedings
of the AAAI Conference on Artificial Intelligence, vol-
ume 39, pp. 3779–3787, 2025b.
Huang, Y., Peng, Z., Zhao, Y., Yang, P., Yang, X., and Shen,
W. Medseg-r: Reasoning segmentation in medical images
with multimodal large language models. arXiv preprint
```
arXiv:2506.10465, 2025c.
```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., and Maier-
Hein, K. H. nnu-net: a self-configuring method for deep
learning-based biomedical image segmentation. Nature
```
methods, 18(2):203–211, 2021.
```
Jaeger, S., Candemir, S., Antani, S., W ´ang, Y.-X. J., Lu,
P.-X., and Thoma, G. Two public chest x-ray datasets for
computer-aided screening of pulmonary diseases. Quan-
```
titative imaging in medicine and surgery, 4(6):475, 2014.
```
Ji, Y., Bai, H., Ge, C., Yang, J., Zhu, Y., Zhang, R., Li, Z.,
Zhanng, L., Ma, W., Wan, X., et al. Amos: A large-scale
abdominal multi-organ benchmark for versatile medical
image segmentation. Advances in Neural Information
Processing Systems, 35:36722–36732, 2022.
Jiang, S., Wang, Y., Song, S., Hu, T., Zhou, C., Pu, B.,
Zhang, Y., Yang, Z., Feng, Y., Zhou, J. T., et al. Hulu-
```
med: A transparent generalist model towards holistic
```
medical vision-language understanding. arXiv preprint
```
arXiv:2510.08668, 2025a.
```
Jiang, Y., Zhang, Y., Zhang, P., Li, Y., Chen, J., Shi, X.,
and Zhen, S. Incentivizing tool-augmented thinking
with images for medical image analysis. arXiv preprint
```
arXiv:2512.14157, 2025b.
```
Jiang, Y., Li, Q., Xu, B., Sun, H., Ding, C., Dong, J., Cai,
Y., Zhang, X., and Yin, J. Ibisagent: Reinforcing pixel-
level visual reasoning in mllms for universal biomedi-
cal object referring and segmentation. arXiv preprint
```
arXiv:2601.03054, 2026.
```
Jieyun, B. and ZhanHong, O. Pubic symphysis-fetal head
segmentation and angle of progression, 2024.
Khaled, R. et al. Categorized digital database for low energy
and subtracted contrast enhanced spectral mammography
images. The Cancer Imaging Archive, 2021.
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C.,
Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo,
W.-Y., et al. Segment anything. In CVPR, pp. 4015–4026,
2023.
Konwer, A., Yang, Z., Bas, E., Xiao, C., Prasanna, P.,
Bhatia, P., and Kass-Hout, T. Enhancing sam with ef-
ficient prompting and preference optimization for semi-
supervised medical image segmentation. In CVPR, pp.
20990–21000, 2025.
Lai, X., Tian, Z., Chen, Y., Li, Y., Yuan, Y., Liu, S., and
Jia, J. Lisa: Reasoning segmentation via large language
model. In CVPR, pp. 9579–9589, 2024.
Lai, X., Li, J., Li, W., Liu, T., Li, T., and Zhao, H. Mini-o3:
Scaling up reasoning patterns and interaction turns for
visual search. arXiv preprint arXiv:2509.07969, 2025a.
Lai, Y., Zhong, J., Li, M., Zhao, S., and Yang, X. Med-
```
r1: Reinforcement learning for generalizable medical
```
reasoning in vision-language models. arXiv preprint
```
arXiv:2503.13939, 2025b.
```
Landman, B., Xu, Z., Igelsias, J., Styner, M., Langerak, T.,
and Klein, A. Miccai multi-atlas labeling beyond the
cranial vault–workshop and challenge. In Proc. MICCAI
Multi-Atlas Labeling Beyond Cranial Vault—Workshop
Challenge, volume 5, pp. 12, 2015.
Lau, J. J., Gayen, S., Ben Abacha, A., and Demner-
Fushman, D. A dataset of clinically generated visual
questions and answers about radiology images. Scientific
```
data, 5(1):1–10, 2018.
```
Li, B., Yan, T., Pan, Y., Luo, J., Ji, R., Ding, J., Xu, Z., Liu,
S., Dong, H., Lin, Z., et al. Mmedagent: Learning to
use medical tools with multi-modal agent. arXiv preprint
```
arXiv:2407.02483, 2024.
```
Li, C., Liu, X., Li, W., Wang, C., Liu, H., Liu, Y., Chen, Z.,
and Yuan, Y. U-kan makes strong backbone for medical
image segmentation and generation. In Proceedings of the
AAAI Conference on Artificial Intelligence, volume 39,
pp. 4652–4660, 2025a.
Li, C.-Y., Chang, K.-J., Yang, C.-F., Wu, H.-Y., Chen, W.,
Bansal, H., Chen, L., Yang, Y.-P., Chen, Y.-C., Chen,
S.-P., et al. Towards a holistic framework for multimodal
llm in 3d brain ct radiology report generation. Nature
```
Communications, 16(1):2258, 2025b.
```
Liu, B., Zhan, L.-M., Xu, L., Ma, L., Yang, Y., and Wu, X.-
M. Slake: A semantically-labeled knowledge-enhanced
dataset for medical visual question answering. In 2021
IEEE 18th international symposium on biomedical imag-
```
ing (ISBI), pp. 1650–1654. IEEE, 2021.
```
10
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
Liu, S., Zheng, B., Chen, W., Peng, Z., Yin, Z., Shao, J., Hu,
J., and Yuan, Y. Endobench: A comprehensive evalua-
tion of multi-modal large language models for endoscopy
analysis, 2025a.
Liu, Y., Peng, B., Zhong, Z., Yue, Z., Lu, F., Yu, B.,
and Jia, J. Seg-zero: Reasoning-chain guided seg-
mentation via cognitive reinforcement. arXiv preprint
```
arXiv:2503.06520, 2025b.
```
Liu, Y., Qu, T., Zhong, Z., Peng, B., Liu, S., Yu, B., and
Jia, J. Visionreasoner: Unified visual perception and
reasoning via reinforcement learning. arXiv preprint
```
arXiv:2505.12081, 2025c.
```
Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H.,
Lin, D., and Wang, J. Visual-rft: Visual reinforcement
fine-tuning. ICCV, 2025d.
Luo, X., Liao, W., Xiao, J., Chen, J., Song, T., Zhang, X.,
Li, K., Metaxas, D. N., Wang, G., and Zhang, S. Word:
A large scale dataset, benchmark and clinical applicable
study for abdominal organ segmentation from ct image.
Medical Image Analysis, 82:102642, 2022.
Ma, J., Zhang, Y., Gu, S., An, X., Wang, Z., Ge, C., Wang,
C., Zhang, F., Wang, Y., Xu, Y., et al. Fast and low-
gpu-memory abdomen ct organ segmentation: the flare
challenge. Medical Image Analysis, 82:102616, 2022.
Ma, J., He, Y., Li, F., Han, L., You, C., and Wang, B. Seg-
ment anything in medical images. Nature Communica-
```
tions, 15(1):654, 2024.
```
Ma, J., Xu, Y., Zhou, F., Wang, Y., Jin, C., Guo, Z., Wu,
J., Tang, O. K., Zhou, H., Wang, X., et al. Pathbench:
A comprehensive comparison benchmark for pathology
foundation models towards precision oncology. arXiv
preprint arXiv:2505.20202, 2025a.
Ma, J., Yang, Z., Kim, S., Chen, B., Baharoon, M., Fallah-
pour, A., Asakereh, R., Lyu, H., and Wang, B. Medsam2:
Segment anything in 3d medical images and videos. arXiv
preprint arXiv:2504.03600, 2025b.
Nath, V., Li, W., Yang, D., Myronenko, A., Zheng, M., Lu,
Y., Liu, Z., Yin, H., Law, Y. M., Tang, Y., et al. Vila-m3:
Enhancing vision-language models with medical expert
knowledge. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pp. 14788–14798, 2025.
Ngoc Lan, P., An, N. S., Hang, D. V., Long, D. V., Trung,
T. Q., Thuy, N. T., and Sang, D. V. Neounet: Towards
accurate colon polyp segmentation and neoplasm detec-
tion. In Advances in visual computing: 16th international
symposium, ISVC 2021, virtual event, October 4-6, 2021,
proceedings, part II, pp. 15–28. Springer, 2021.
Ning, J., Li, W., Tang, C., Lin, J., Ma, C., Zhang, C., Liu,
J., Chen, Y., Gao, S., Liu, L., et al. Unimedvl: Unify-
ing medical multimodal understanding and generation
through observation-knowledge-analysis. arXiv preprint
```
arXiv:2510.15710, 2025.
```
OpenAI. Thinking with images, 2025.
URL https://openai.com/index/
thinking-with-images/.
Orlando, J. I., Fu, H., Breda, J. B., Van Keer, K., Bathula,
D. R., Diaz-Pinto, A., Fang, R., Heng, P.-A., Kim, J.,
Lee, J., et al. Refuge challenge: A unified framework for
evaluating automated methods for glaucoma assessment
from fundus photographs. Medical Image Analysis, 59:
101570, 2020.
Pan, J., Liu, C., Wu, J., Liu, F., Zhu, J., Li, H. B., Chen, C.,
Ouyang, C., and Rueckert, D. Medvlm-r1: Incentivizing
medical reasoning capability of vision-language models
```
(vlms) via reinforcement learning. In MICCAI, pp. 337–
```
347. Springer, 2025.
Peng, Y., Zhang, G., Zhang, M., You, Z., Liu, J., Zhu, Q.,
Yang, K., Xu, X., Geng, X., and Yang, X. Lmm-r1: Em-
powering 3b lmms with strong reasoning abilities through
two-stage rule-based rl. arXiv preprint arXiv:2503.07536,
2025.
Rasheed, H., Maaz, M., Shaji, S., Shaker, A., Khan, S.,
Cholakkal, H., Anwer, R. M., Xing, E., Yang, M.-H., and
Khan, F. S. Glamm: Pixel grounding large multimodal
model. In CVPR, pp. 13009–13018, 2024.
Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T.,
Khedr, H., R ¨adle, R., Rolland, C., Gustafson, L., et al.
Sam 2: Segment anything in images and videos. arXiv
preprint arXiv:2408.00714, 2024.
Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolu-
tional networks for biomedical image segmentation. In
MICCAI, pp. 234–241. Springer, 2015.
Ru, J., Yan, S., Yin, Y., Zou, Y., and Ge, Z. Der-
```
mogpt: Open weights and open data for morphology-
```
grounded dermatological reasoning mllms. arXiv preprint
```
arXiv:2601.01868, 2026.
```
Schulman, J., Wolski, F., Dhariwal, P., Radford, A.,
and Klimov, O. Proximal policy optimization algo-
rithms, 2017. URL https://arxiv.org/abs/
1707.06347.
Sellergren, A., Kazemzadeh, S., Jaroensri, T., Kiraly, A.,
Traverse, M., Kohlberger, T., Xu, S., Jamil, F., Hughes,
C., Lau, C., et al. Medgemma technical report. arXiv
preprint arXiv:2507.05201, 2025.
11
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang,
H., Zhang, M., Li, Y., et al. Deepseekmath: Pushing
the limits of mathematical reasoning in open language
models. arXiv preprint arXiv:2402.03300, 2024.
Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q.,
Zhang, Z., Zhao, K., Zhang, Q., et al. Vlm-r1: A stable
and generalizable r1-style large vision-language model.
arXiv preprint arXiv:2504.07615, 2025a.
Shen, H., Zhao, K., Zhao, T., Xu, R., Zhang, Z., Zhu, M.,
and Yin, J. Zoomeye: Enhancing multimodal llms with
human-like zooming capabilities through tree-based im-
age exploration. In EMNLP, pp. 6613–6629, 2025b.
Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang,
R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexi-
ble and efficient rlhf framework. arXiv preprint arXiv:
2409.19256, 2024.
Su, A., Wang, H., Ren, W., Lin, F., and Chen, W.
Pixel reasoner: Incentivizing pixel-space reasoning with
curiosity-driven reinforcement learning. arXiv preprint
```
arXiv:2505.15966, 2025a.
```
Su, Z., Li, L., Song, M., Hao, Y., Yang, Z., Zhang, J., Chen,
G., Gu, J., Li, J., Qu, X., et al. Openthinkimg: Learning to
think with images via visual tool reinforcement learning.
arXiv preprint arXiv:2505.08617, 2025b.
Vitale, S., Orlando, J. I., Iarussi, E., and Larrabide, I. Im-
proving realism in patient-specific abdominal ultrasound
simulation using cyclegans. International Journal of
```
Computer Assisted Radiology and Surgery, 15(2):183–
```
192, 2020.
Wang, G., Zhao, J., Liu, X., Liu, Y., Cao, X., Li, C., Liu,
Z., Sun, Q., Zhou, F., Xing, H., et al. Citrus-v: Ad-
vancing medical foundation models with unified medical
image grounding for clinical reasoning. arXiv preprint
```
arXiv:2509.19090, 2025a.
```
Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen,
J., Su, Y., Huang, Z., Shen, Y., Fu, B., Zhang, S., He, J.,
and Qiao, Y. Sam-med3d: Towards general-purpose seg-
mentation models for volumetric medical images, 2024.
URL https://arxiv.org/abs/2310.15161.
Wang, L., Wang, H., Yang, H., Mao, J., Yang, Z., Shen,
J., and Li, X. Interpretable bilingual multimodal large
language model for diverse biomedical tasks. In Interna-
tional Conference on Learning Representations, 2025b.
Wang, W., Ma, Z., Ding, M., Zheng, S., Liu, S., Liu, J.,
Ji, J., Chen, W., Li, X., Shen, L., and Yuan, Y. Medical
reasoning in the era of llms: A systematic review of
enhancement techniques and applications, 2025c. URL
```
https://arxiv.org/abs/2508.00669.
```
Wang, Z., Wu, J., Cai, L., Low, C. H., Yang, X., Li, Q., and
Jin, Y. Medagent-pro: Towards evidence-based multi-
modal medical diagnosis via reasoning agentic workflow.
arXiv preprint arXiv:2503.18968, 2025d.
Wei, C., Zhong, Y., Tan, H., Liu, Y., Zhao, Z., Hu, J.,
and Yang, Y. Hyperseg: Towards universal visual seg-
mentation with large language model, 2024a. URL
```
https://arxiv.org/abs/2411.17606.
```
Wei, X., Cao, J., Jin, Y., Lu, M., Wang, G., and Zhang, S.
I-medsam: Implicit medical image segmentation with seg-
ment anything. In ECCV, pp. 90–107. Springer, 2024b.
Wu, J., Wang, Z., Hong, M., Ji, W., Fu, H., Xu, Y., Xu,
M., and Jin, Y. Medical sam adapter: Adapting segment
anything model for medical image segmentation. Medical
image analysis, 102:103547, 2025a.
Wu, L., Nie, Y., He, S., Zhuang, J., Luo, L., Mahboobani, N.,
Vardhanabhuti, V., Chan, R. C. K., Peng, Y., Rajpurkar,
P., et al. Unibiomed: A universal foundation model for
grounded biomedical image interpretation. arXiv preprint
```
arXiv:2504.21336, 2025b.
```
Wu, M., Yang, J., Jiang, J., Li, M., Yan, K., Yu, H., Zhang,
M., Zhai, C., and Nahrstedt, K. Vtool-r1: Vlms learn to
think with images via reinforcement learning on multi-
modal tool use, 2025c. URL https://arxiv.org/
abs/2505.19255.
Xu, H., Nie, Y., Wang, H., Chen, Y., Li, W., Ning, J., Liu,
L., Wang, H., Zhu, L., Liu, J., et al. Medground-r1:
Advancing medical image grounding via spatial-semantic
rewarded group relative policy optimization. In MICCAI,
pp. 391–401. Springer, 2025a.
Xu, W., Chan, H. P., Li, L., Aljunied, M., Yuan, R., Wang,
J., Xiao, C., Chen, G., Liu, C., Li, Z., et al. Lingshu:
A generalist foundation model for unified multimodal
medical understanding and reasoning. arXiv preprint
```
arXiv:2506.07044, 2025b.
```
Yan, Z., Diao, M., Yang, Y., Xu, J., Zhang, K., Jing, R.,
Yang, L., Liu, Y., Liang, K., and Ma, Z. Medreasoner:
Reinforcement learning drives reasoning grounding from
clinical thought to pixel-level precision. arXiv preprint
```
arXiv:2508.08177, 2025.
```
Ye, J., Wang, G., Li, Y., Deng, Z., Li, W., Li, T., Duan,
H., Huang, Z., Su, Y., Wang, B., et al. Gmai-mmbench:
A comprehensive multimodal evaluation benchmark to-
wards general medical ai. Advances in Neural Informa-
tion Processing Systems, 37:94327–94427, 2024.
Ye, Y., Xie, Y., Zhang, J., Chen, Z., and Xia, Y. Uniseg: A
prompt-driven universal segmentation model as well as a
strong representation learner. In MICCAI, pp. 508–518.
Springer, 2023.
12
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
You, Z. and Wu, Z. Seg-r1: Segmentation can be surpris-
ingly simple with reinforcement learning. arXiv preprint
```
arXiv:2506.22624, 2025.
```
Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai,
W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source
llm reinforcement learning system at scale. arXiv preprint
```
arXiv:2503.14476, 2025.
```
Zambrano Chaves, J. M., Huang, S.-C., Xu, Y., Xu, H.,
Usuyama, N., Zhang, S., Wang, F., Xie, Y., Khademi, M.,
Yang, Z., et al. A clinically accessible small multimodal
radiology model and evaluation metric for chest x-ray
```
findings. Nature Communications, 16(1):3108, 2025.
```
Zhang, H., Gu, X., Li, J., Ma, C., Bai, S., Zhang, C., Zhang,
B., Zhou, Z., He, D., and Tang, Y. Thinking with videos:
Multimodal tool-augmented reinforcement learning for
long video reasoning. arXiv preprint arXiv:2508.04416,
2025a.
Zhang, J., Kim, J., O’Donoghue, B., and Boyd, S. Sam-
ple efficient reinforcement learning with reinforce, 2020.
URL https://arxiv.org/abs/2010.11364.
Zhang, S., Liu, Q., Qin, G., Naumann, T., and Poon,
H. Med-rlvr: Emerging medical reasoning from a 3b
base model via reinforcement learning. arXiv preprint
```
arXiv:2502.19655, 2025b.
```
Zhang, S., Zhang, Q., Zhang, S., Liu, X., Yue, J., Lu, M.,
Xu, H., Yao, J., Wei, X., Cao, J., et al. A generalist
foundation model and database for open-world medical
image segmentation. Nature Biomedical Engineering, pp.
1–16, 2025c.
Zhang, Y.-F., Lu, X., Yin, S., Fu, C., Chen, W., Hu, X., Wen,
B., Jiang, K., Liu, C., Zhang, T., Fan, H., Chen, K., Chen,
J., Ding, H., Tang, K., Zhang, Z., Wang, L., Yang, F., Gao,
T., and Zhou, G. Thyme: Think beyond images, 2025d.
URL https://arxiv.org/abs/2508.11630.
Zhao, T., Gu, Y., Yang, J., Usuyama, N., Lee, H. H., Kiblawi,
S., Naumann, T., Gao, J., Crabtree, A., Abel, J., et al. A
foundation model for joint segmentation, detection and
recognition of biomedical objects across nine modalities.
```
Nature methods, 22(1):166–176, 2025.
```
Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C.,
Dang, K., Liu, Y., Men, R., Yang, A., Zhou, J., and Lin,
J. Group sequence policy optimization, 2025a. URL
```
https://arxiv.org/abs/2507.18071.
```
Zheng, L., Yin, L., Xie, Z., Sun, C. L., Huang, J., Yu, C. H.,
Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., et al.
```
Sglang: Efficient execution of structured language model
```
programs. Advances in neural information processing
systems, 37:62557–62583, 2024a.
Zheng, Y., Zhang, R., Zhang, J., YeYanhan, Y., and Luo,
Z. Llamafactory: Unified efficient fine-tuning of 100+
language models. In ACL, pp. 400–410, 2024b.
Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L.,
Shen, C., and Yu, X. Deepeyes: Incentivizing” thinking
with images” via reinforcement learning. arXiv preprint
```
arXiv:2505.14362, 2025b.
```
Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., and
Liang, J. Unet++: A nested u-net architecture for medical
image segmentation. In International workshop on deep
learning in medical image analysis, pp. 3–11. Springer,
2018.
Zhu, M., Tian, Y., Chen, H., Zhou, C., Guo, Q., Liu, Y.,
Yang, M., and Shen, C. Segagent: Exploring pixel un-
derstanding capabilities in mllms by imitating human
annotator trajectories. In CVPR, pp. 3686–3696, 2025.
13
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
Supplementary
A. Datasets
In our experiments, we utilize a comprehensive collection of 21 open-source datasets spanning 6 modalities, including CT
```
(FLARE22 (Ma et al., 2022), KiTS (Heller et al., 2023), LIDC-IDRI (Armato III et al., 2011), BTCV (Landman et al.,
```
```
2015), AMOS-CT (Ji et al., 2022), and WORD (Luo et al., 2022)), MRI (ACDC (Bernard et al., 2018), LGG (Buda et al.,
```
```
2019), and AMOS-MRI (Ji et al., 2022)), X-Ray (CXRMask (Jaeger et al., 2014), Radiography series (Chowdhury et al.,
```
```
2020), and CDD-CESM (Khaled et al., 2021)), Ultrasound (BreastUS (Vitale et al., 2020), LiverUS (Vitale et al., 2020), and
```
```
FH-PS-AOP (Jieyun & ZhanHong, 2024)), Fundus (REFUGE (Orlando et al., 2020)), and Endoscopy (NeoPolyp (Ngoc Lan
```
```
et al., 2021) and PolypGen (Ali et al., 2023)). All data formats are standardized following previous methods (Zhao et al.,
```
```
2025; Wu et al., 2025b).
```
```
Table 4. Descriptions of datasets used in this work, including modalities, regions of interest, and the number of triplets (image-text-label).
```
Dataset Modality Regions of interest Number
```
FLARE22 (Ma et al., 2022) CT Abdomen organs 26,802
```
```
KiTS (Heller et al., 2023) CT Kidney & Kidney Tumor 44,557
```
```
LIDC-IDRI (Armato III et al., 2011) CT Lung nodule 9,122
```
```
BTCV (Landman et al., 2015) CT Abdomen organs 12,176
```
```
AMOS-CT (Ji et al., 2022) CT Abdomen organs 138,371
```
```
WORD (Luo et al., 2022) CT Abdomen organs 58,898
```
```
ACDC (Bernard et al., 2018) MRI Heart 7,666
```
```
LGG (Buda et al., 2019) MRI Brain Tumor 2,542
```
```
AMOS-MRI (Ji et al., 2022) MRI Abdomen 52,625
```
```
CXRMask (Jaeger et al., 2014) X-Ray Chest 1,698
```
```
Radiography-Lung-opacity (Chowdhury et al., 2020) X-Ray Chest 6,012
```
```
Radiography-Normal (Chowdhury et al., 2020) X-Ray Chest 30,574
```
```
Radiography-Viral-Pneumonia (Chowdhury et al., 2020) X-Ray Chest 1,345
```
```
Radiography-COVID (Chowdhury et al., 2020) X-Ray Chest 10,844
```
```
CDD-CESM (Khaled et al., 2021) X-Ray Breast lesion 1,233
```
```
BreastUS (Vitale et al., 2020) Ultrasound Breast lesion 1,294
```
```
LiverUS (Vitale et al., 2020) Ultrasound Liver 39
```
```
FH-PS-AOP (Jieyun & ZhanHong, 2024) Ultrasound Transperineal 8,000
```
```
REFUGE (Orlando et al., 2020) Fundus Retinal 2,400
```
```
NeoPolyp (Ngoc Lan et al., 2021) Endoscopy Colon polyp 2,050
```
```
PolypGen (Ali et al., 2023) Endoscopy Colon polyp 1,411
```
Total 6 Modalities 419,659
B. Implementation Details
B.1. Trajectory Construction
To train our model for a multi-step medical image segmentation agent, we require a dataset of expert-like interaction
trajectories. We employed an automated algorithm to generate these trajectories by simulating the sequential refinement
```
process an expert annotator would perform. While existing methods (Zhu et al., 2025; Jiang et al., 2026) rely on a rigid
```
point-wise simulation driven by pixel-level discrepancies, they typically restrict the action space to sequential clicks. Such
a limitation fails to capture the multi-modal nature of human workflows, where practitioners often initiate segmentation
by defining a Bounding Box before refining with precise clicks. To better align with human intuition, we propose a
hybrid prompting strategy, as detailed in Algorithm 1. This strategy encompasses both Box-to-Point and Sequential-Click
paradigms to generate diverse and realistic interaction trajectories.
14
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
Algorithm 1 Hybrid Prompting Strategy for Trajectory Generation
```
Require: Image I, ground truth mask Mtarget, state s, segmentation model fθ , max clicks K, gain threshold τ ,
```
max retries N
```
Ensure: Trajectory T = {(a0, M0), (a1, M1), . . . , (aT , MT )}
```
```
1: paradigm ← {“Box-to-Point”, “Sequential-Click”}
```
2: T ← ∅
```
3: {Initialization at t = 0 using either Bounding Box or Centroid Click}
```
4: if paradigm = “Box-to-Point” then
```
5: a0 ← ExtractBoundingBox(Mtarget) + ϵ, where ϵ ∼ U(−δ, δ)
```
6: else
```
7: a0 ← 1|Mtarget|P(x,y)∈Mtarget (x, y) + ϵ
```
8: end if
```
9: M0 ← fθ (I, a0)
```
```
10: T ← T ∪ {(a0, M0)}
```
11: for t = 1 to K do
12: FNt ← Mtarget \ Mt−1, FPt ← Mt−1 \ Mtarget
13: if |FNt| = 0 and |FPt| = 0 then
14: break
15: end if
```
16: {Error-driven refinement with retry mechanism}
```
17: success ← False
18: for trial = 1 to N do
19: if |FNt| > |FPt| then
```
20: at ← arg maxp∈FNt D(p) (Positive Click)
```
21: else
```
22: at ← arg maxp∈FPt D(p) (Negative Click)
```
23: end if
```
24: Mt ← fθ (I, st−1)
```
```
25: ∆IoUt ← IoU(Mt, Mtarget) − IoU(Mt−1, Mtarget)
```
26: if ∆IoUt ≥ τ then
```
27: T ← T ∪ {(at, Mt)}
```
28: success ← True
29: break
30: else
```
31: Resample at by selecting alternative local maxima in D(·)
```
32: end if
33: end for
34: if not success then
```
35: break {Early stop if no significant improvement after N retries}
```
36: end if
37: end for
38: return T
The key innovation of our approach lies in an adaptive error-driven refinement mechanism combined with progress-
constrained sampling. Rather than randomly sampling prompts, we leverage morphological analysis of prediction errors.
```
We identify False Negative (FN) regions where the model under-segments and False Positive (FP) regions where it over-
```
segments, then apply distance transforms to localize the most significant error clusters. This ensures that each corrective
action addresses the most salient morphological defects. Furthermore, to guarantee trajectory quality, we enforce a constraint
where each action must yield a measurable IoU improvement, filtering out ineffective interactions through an iterative retry
mechanism. Our trajectory generation strategy differs primarily in the initial prompt a0. In the Box-to-Point workflow, a0 is
```
a bounding box generated by extracting the axis-aligned rectangle of Mtarget with a random jitter ϵ ∼ U(−δ, δ) to simulate
```
human imprecision. In contrast, the Sequential-Click paradigm begins with a point prompt sampled from the centroid of the
target mask:
15
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
```
a0 = 1|M
```
target|
X
```
(x,y)∈Mtarget
```
```
(x, y) + ϵ (8)
```
Regardless of initialization, all subsequent refinement steps at t > 0 follow a unified error-driven mechanism. The agent
```
analyzes the prediction error Et = M (t)pred ⊕ Mtarget, decomposed into FNt = Mtarget \ M (t)pred and FPt = M (t)pred \ Mtarget. By
```
```
applying a distance transform D(·), we identify the centroids of the largest error components. Corrective clicks are sampled
```
```
as:
```
```
at =
```
```
(
```
```
arg maxp∈FNt D(p), if |FNt| > |FPt|
```
```
arg maxp∈FPt D(p), otherwise (9)
```
with positive labels for FN clicks and negative labels for FP clicks.To ensure the efficiency of synthesized trajectories, we
implement a progress-constrained sampling mechanism. We require each simulated action to yield an incremental IoU gain
```
(∆IoU) exceeding a predefined threshold τ :
```
```
∆IoUt = IoU(Mt, Mtarget) − IoU(Mt−1, Mtarget) ≥ τ (10)
```
where τ is typically set to 0.04. If a candidate action fails to satisfy this threshold, the simulator performs iterative resampling
```
up to N trials (default N = 5) to identify a more constructive interaction. During each retry, we re-sample the click position
```
within the error region by selecting alternative local maxima in the distance transform map or introducing controlled
perturbations. If no valid action is found after N trials, the trajectory generation is terminated early. This validation process
ensures that the trajectory dataset Dtraj is composed of high-quality, monotonically improving sequences. We set K = 5
as the maximum number of refinement clicks, τ = 0.04 as the gain threshold, and N = 5 as the maximum retry attempts.
```
Bounding box jitter is sampled from U(−5, 5) pixels, while click jitter follows N (0, 22) pixels. The generated trajectories
```
are stored in JSON format, recording each action at, intermediate mask Mt.
```
Our implementation leverages MedSAM2 (Ma et al., 2025b) and IMISNet (Cheng et al., 2025) as the base segmentation
```
models fθ . Based on this trajectory generation pipeline, we initially generated 334,616 trajectories for both the Box-to-Point
```
and Sequential-Click strategies using MedSAM2. To ensure the quality of the Supervised Fine-Tuning (SFT) data, we
```
applied an IoU threshold of 0.7 to filter out low-performing samples. This resulted in 188,687 click-based and 260,446
box-based trajectories, totaling 449,133 high-quality samples. A similar procedure was applied using IMISNet, yielding
235,993 click trajectories and 283,522 box trajectories, for a total of 519,515 samples. These filtered trajectories served as
the SFT data to train their respective models.
B.2. Prompt Design
The reasoning capabilities of MedSAM-Agent are governed by a structured system prompt that defines the agent’s role,
tool-use protocols, and decision-making heuristics. This prompt ensures that the MLLM interprets the medical image not as
a static object, but as a dynamic environment requiring strategic intervention.
B.2.1. TOOL SPECIFICATIONS
We provide the agent with a rigorous schema for tool calling. Specifically, the model is instructed to output its decisions
in a standard JSON format, selecting from add box, add point, or stop action. To maintain procedural clarity,
the system prompt enforces a “Single Action per Turn” rule, requiring the model to observe the visual feedback from the
segmentation backend before initiating the subsequent refinement step.
B.2.2. SYSTEM PROMPT
As for the system prompt, we first initialize the model as a professional segmentation annotator, grounding its objective
in high-precision clinical mask creation. The prompt explicitly informs the agent that the mask is rendered as a “semi-
transparent green overlay”, enabling the model to visually contrast the current prediction with the underlying anatomy to
identify areas requiring refinement.
16
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
```
Functional Tools (json)
```
[
```
{
```
"type": "function",
```
"function": {
```
"name": "add_bbox",
"description": "Add a bounding box to initialize or refine the segmentation.",
```
"parameters": {
```
"type": "object",
```
"properties": {
```
```
"bbox_2d": { "type": "array", "items": {"type": "integer"}, "minItems": 4,
```
"maxItems": 4, "description": "2D bounding box in [x1, y1, x2, y2]
```
format" }
```
```
},
```
"required": ["bbox_2d"]
```
}
```
```
}
```
```
},
```
```
{
```
"type": "function",
```
"function": {
```
"name": "add_point",
```
"description": "Add a point to refine the mask (positive to include areas,
```
```
negative to exclude areas).",
```
```
"parameters": {
```
"type": "object",
```
"properties": {
```
```
"point_2d": { "type": "array", "items": {"type": "integer"}, "minItems": 2,
```
```
"maxItems": 2, "description": "2D coordinate point in [x, y] format" },
```
```
"point_type": { "type": "string", "enum": ["positive", "negative"] }
```
```
},
```
"required": ["point_2d", "point_type"]
```
}
```
```
}
```
```
},
```
```
{
```
"type": "function",
```
"function": {
```
"name": "stop_action",
"description": "Stop the refinement process when the mask accurately covers the
target object.",
```
"parameters": { "type": "object", "properties": {} }
```
```
}
```
```
}
```
]
System Prompt
You are a professional segmentation annotator specializing in mask creation and refinement. Your core task is to
segment the USER-SPECIFIED TARGET REGION from the provided image. No preliminary mask is available—you
must first create an initial mask using the tool, then iteratively refine it to achieve pixel-level accuracy. The mask will
```
be displayed as a semi-transparent green overlay; your goal is to ensure it exactly covers the entire target region and
```
```
excludes all non-target areas (e.g., background, adjacent objects).
```
# Tools
You must call one function to assist with the user query. You are provided with function signatures within
<tools></tools> XML tags:
<tools>
[The JSON-formatted tool signatures are inserted here]
</tools>
17
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
For each function call, return a json object with function name and arguments within <tool call></tool call>
XML tags:
<tool call>
```
{"name": <function-name>, "arguments": <args-json-object>}
```
</tool call>
Only use the provided functions to complete your task. Do not invent or assume any other functions. Carefully
consider the current mask state before each action.
B.2.3. INTERACTION PROTOCOL
The interaction protocol is structured as a closed-loop dialogue that transitions from initial localization to iterative refinement,
mimicking a clinical workflow. The process begins with an initialization turn, where the agent receives the target specification
and performs the first localization action. This is followed by subsequent refinement turns, during which the agent evaluates
the visual feedback of the mask overlay against the anatomical structure, strategically placing positive or negative points
to correct errors. The cycle continues until the agent, guided by its internal reward-optimized policy, determines that the
segmentation has reached pixel-level accuracy and invokes the termination action to finalize the task efficiently.
Initial Turn prompt
```
<image> The target to be segmented is: {target description}.
```
Now, please analyze the original image, then decide your first action.
Subsequent Turns prompt
<image> Here is the updated mask after your previous action. Based on this, what is your next action? If the mask is
now accurate, you can call ’stop action’ to finish.
```
Based on these prompts, we format the resulting interaction sequences into a standard Supervised Fine-Tuning (SFT) data
```
structure to facilitate the model’s cold-start training.
B.3. Training Details
The resolution of all datasets in our experiments is 1024 × 1024. During the construction of interaction trajectories, we map
```
the spatial coordinates to absolute integers within a [0, 1000] range to ensure compatibility with the Qwen3-VL (Bai et al.,
```
```
2025) architecture. This normalization enables the model to effectively process spatial prompts and achieve high-precision
```
interactive segmentation.
```
During the SFT stage, we utilize the Llama-Factory (Zheng et al., 2024b) framework with a learning rate of 1 × 10−5 and a
```
total batch size of 64. The number of training epochs is 4. To maintain training efficiency and leverage pre-trained visual
representations, we only update the parameters of the LLM backbone, while keeping the vision encoder and projector frozen.
The training is accelerated using the DeepSpeed ZeRO-3 strategy.
```
The RL stage is implemented via the Verl framework (Sheng et al., 2024), using a learning rate of 1 × 10−5, a batch size of 8,
```
and a sampling size of 8 per prompt. The maximum interaction depth is set to 5 turns. To maximize training throughput, we
```
employ an asynchronous multi-turn sampling mechanism integrated with the SGLang (Zheng et al., 2024a) runtime, enabling
```
efficient concurrent inference during the policy rollout phase. As for the hyper-parameters, we set wiou = wdice = 0.5,
```
w1 = 0.2, w2 = 0.8, λ1 = 0.1, λ2 = 1.0, and λ3 = 0.01. All experiments were conducted on 8 × NVIDIA H20 (96G)
```
GPUs.
During the RL stage, given that medical segmentation requires a rigorous focus on fine-grained visual details, we deviate
from the conventional reasoning-agent setup by omitting a dedicated <think> token. This design choice is driven by
```
two primary objectives: first, to significantly accelerate inference speed by reducing token overhead; and second, to force
```
the model to prioritize dense image-level features over textual deliberation. By bypassing the explicit thinking step, we
encourage the agent to internalize spatial reasoning directly within its action space, ensuring that its decision-making is
more tightly coupled with the immediate morphological characteristics of the lesion.
18
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
C. Experiments Results
C.1. Comparison Experiment
In our experiments, we compare our method with representative interactive segmentation frameworks, including the general-
```
purpose SAM2 (Ravi et al., 2024) and domain-specific models such as MedSAM2 (Ma et al., 2025b) and IMISNet (Cheng
```
```
et al., 2025). For these baselines, we report results using both Point and Box prompts. “Point” results are generated
```
using the center of the ground-truth mask, while “Box” results utilize corresponding bounding boxes. The Box prompt
performance is generally regarded as the empirical upper bound for single-turn interactive segmentation. Furthermore,
```
we benchmark against SOTA MLLM-based methods, including four general-domain models (LISA (Lai et al., 2024),
```
```
GALLM (Rasheed et al., 2024), HyperSeg (Wei et al., 2024a), and Seg-R1 (You & Wu, 2025)) and three specialized medical
```
```
MLLMs (MedPLIB (Huang et al., 2025b), UniBioMed (Wu et al., 2025b), and Citrus-V (Wang et al., 2025a)).
```
Table 5 and Table 6 demonstrate all results of 21 datasets across 6 medical modalities. The empirical results reveal that the
performance of our autonomous agent is intrinsically linked to the foundational capabilities of the underlying segmentation
model, yet it demonstrates a unique capacity to improve upon baseline results through strategic interaction. For instance,
while specialized models like IMISNet demonstrate exceptional proficiency in CT datasets, their performance significantly
```
decreases when applied to X-Ray, as observed in the Radiography (Chowdhury et al., 2020) results. However, our multi-turn
```
agentic framework addresses these limitations by iteratively identifying residual errors and applying corrective logic. By
engaging in multiple rounds of interaction, the agent achieves performance that surpasses the inherent limits of a single-turn
approach, thereby demonstrating the fundamental significance of multi-turn strategic engagement. This phenomenon
suggests that MedSAM-Agent has internalized a sophisticated refinement logic that goes beyond simple tool invocation.
By evaluating the segmentation state at each turn, the agent autonomously decides on corrective measures to eliminate
hallucinations and under-segmentation. This active perception allows the agent to enhance raw model capabilities to reach
the high precision required for clinical diagnostics, proving that autonomous, multi-turn strategic interaction is superior to
passive, single-turn prompting paradigms.
C.2. Case Study
In this section, we provide qualitative examples of MedSAM-Agent across various medical imaging modalities to demonstrate
```
its interactive reasoning capabilities (Fig. 4). Unlike traditional single-turn models, our agent engages in a multi-step
```
refinement process, iteratively evaluating the current segmentation mask and adjusting its strategy based on visual feedback.
In these visualizations, yellow prompts indicate positive clicks or bounding boxes used to define the target anatomy, while
red prompts represent negative clicks deployed to exclude false-positive regions or refine over-segmented boundaries.
As shown in the sequential turns, MedSAM-Agent typically initiates the process with a global constraint and subsequently
applies local corrective measures to eliminate hallucinations and fill in under-segmented areas. This autonomous refinement
loop effectively mimics a clinician’s iterative workflow, proving that the model can successfully internalize sophisticated
decision heuristics to achieve superior mask fidelity in challenging clinical environments.
D. Future Works
In future research, we plan to extend the MedSAM-Agent framework along the following three strategic dimensions:
Expansion to Volumetric Modalities. While the current study focuses on 2D image slices, many clinical diagnostic tasks
rely on 3D volumetric data, such as CT and MRI scans. We aim to extend the agent’s action space and reinforcement
learning environment to handle 3D spatial contexts. This transition will enable the agent to capture cross-slice anatomical
continuity, providing more consistent and clinically accurate volumetric segmentations.
Development of a Unified multi-modal Agent. While our work focuses on the specialized task of interactive segmentation,
the proposed autonomous decision-making framework is inherently extensible. In the future, we intend to evolve MedSAM-
```
Agent into a unified medical AI assistant capable of performing diverse medical imaging tasks (e.g., Medical VQA, lesion
```
```
classification, and automated report generation) within a single, cohesive architecture. By treating these distinct tasks as
```
specialized tool-use actions, we can integrate new capabilities without altering the core agentic framework. This evolution
aims to provide a comprehensive assistant that supports the entire clinical workflow, seamlessly transitioning between
perception, reasoning, and high-precision interaction.
Enhancement of Computational Efficiency. A potential limitation of multi-turn iterative paradigms is the increased
19
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
cumulative latency during inference. In future work, we plan to explore more efficient architectures and sampling strategies
to mitigate this overhead. Specifically, we will investigate the integration of early-exit mechanisms to terminate redundant
computation when high-confidence masks are achieved and leverage speculative decoding or KV-cache optimization
techniques tailored for multi-modal agents. These advancements will ensure that the superior precision of MedSAM-Agent
is delivered with the near-real-time responsiveness required for high-throughput clinical applications.
20
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
Table 5. Quantitative comparison across CT, MRI, and Endoscopy. Point and Box denote the interactive prompts derived from the
ground-truth center points and bounding boxes, respectively, which are utilized to evaluate the single-round segmentation performance.
Bold and underlined values represent the best and second-best performance among non-interactive methods, respectively.
CT MRI Endoscopy
Flare22 KiTS LIDC-IDRI BTCV amos-CT WORD ACDC LGG amos-MRI NeoPolyp PolypGen
SAM2-Point Dice 0.766 0.666 0.508 0.634 0.601 0.487 0.532 0.439 0.492 0.641 0.678IoU 0.697 0.563 0.430 0.564 0.529 0.405 0.446 0.354 0.436 0.585 0.608
SAM2-Box Dice 0.920 0.818 0.756 0.890 0.869 0.772 0.809 0.865 0.839 0.934 0.911IoU 0.863 0.720 0.648 0.821 0.796 0.674 0.718 0.775 0.752 0.887 0.856
MedSAM-Point Dice 0.831 0.655 0.763 0.795 0.751 0.686 0.556 0.869 0.577 0.844 0.800IoU 0.746 0.528 0.645 0.700 0.660 0.558 0.498 0.779 0.476 0.762 0.711
MedSAM-Box Dice 0.913 0.836 0.773 0.890 0.888 0.781 0.847 0.893 0.821 0.933 0.916IoU 0.847 0.735 0.668 0.815 0.814 0.674 0.766 0.813 0.725 0.884 0.859
IMISNet-Point Dice 0.845 0.677 0.268 0.819 0.745 0.692 0.753 0.622 0.660 0.798 0.764IoU 0.766 0.554 0.221 0.725 0.638 0.577 0.634 0.485 0.548 0.704 0.659
IMISNet-Box Dice 0.911 0.826 0.524 0.870 0.869 0.829 0.857 0.860 0.830 0.924 0.885IoU 0.852 0.740 0.443 0.797 0.794 0.736 0.767 0.766 0.741 0.871 0.821
LISA Dice 0.116 0.147 0.006 0.062 0.097 0.098 0.069 0.185 0.076 0.271 0.293IoU 0.070 0.086 0.003 0.036 0.057 0.055 0.037 0.107 0.044 0.220 0.225
GLAMM Dice 0.128 0.110 0.005 0.078 0.106 0.103 0.050 0.188 0.078 0.204 0.285IoU 0.079 0.061 0.003 0.048 0.066 0.059 0.027 0.109 0.046 0.149 0.208
HyperSeg Dice 0.117 0.175 0.006 0.068 0.099 0.121 0.160 0.183 0.117 0.392 0.400IoU 0.089 0.129 0.003 0.046 0.067 0.078 0.102 0.119 0.083 0.344 0.341
Seg-R1 Dice 0.113 0.159 0.009 0.087 0.102 0.124 0.134 0.168 0.076 0.515 0.551IoU 0.080 0.104 0.005 0.055 0.064 0.076 0.081 0.098 0.045 0.457 0.485
MedPLIB Dice 0.046 0.033 0.001 0.083 0.054 0.096 0.118 0.318 0.096 0.123 0.142IoU 0.035 0.021 0.001 0.058 0.038 0.075 0.088 0.252 0.070 0.092 0.102
UniBiomed Dice 0.825 0.755 0.502 0.735 0.822 0.707 0.827 0.863 0.732 0.851 0.705IoU 0.747 0.670 0.405 0.646 0.738 0.599 0.738 0.776 0.635 0.784 0.623
Citrus-V Dice 0.480 0.470 0.221 0.428 0.262 0.314 0.438 0.305 0.235 0.549 0.783IoU 0.417 0.377 0.170 0.370 0.220 0.258 0.326 0.254 0.194 0.510 0.722
Ours-IMISNet Dice 0.848 0.791 0.406 0.780 0.828 0.741 0.836 0.828 0.685 0.813 0.793IoU 0.779 0.710 0.340 0.702 0.751 0.642 0.743 0.728 0.583 0.753 0.719
Ours-MedSAM2 Dice 0.836 0.703 0.523 0.773 0.805 0.681 0.816 0.844 0.719 0.814 0.809IoU 0.761 0.597 0.448 0.690 0.723 0.579 0.726 0.755 0.623 0.753 0.735
21
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
Table 6. Quantitative comparison across X-Ray, Ultrasound, and Fundus. Point and Box denote the interactive prompts derived from
the ground-truth center points and bounding boxes, respectively, which are utilized to evaluate the single-round segmentation performance.
Bold and underlined values represent the best and second-best performance among non-interactive methods, respectively.
X-Ray Ultrasound Fundus
CXRMask Rad-LO Rad-N Rad-VP Rad-COVID CDD-CESM BreastUS LiverUS FH-PS-AOP REFUGE
SAM2-Point Dice 0.647 0.546 0.627 0.536 0.582 0.339 0.685 0.557 0.341 0.362IoU 0.533 0.409 0.499 0.392 0.454 0.246 0.582 0.424 0.217 0.265
SAM2-Box Dice 0.940 0.790 0.928 0.783 0.890 0.628 0.889 0.849 0.876 0.717IoU 0.890 0.667 0.869 0.667 0.809 0.497 0.811 0.745 0.782 0.574
MedSAM-Point Dice 0.887 0.878 0.956 0.922 0.941 0.477 0.808 0.421 0.543 0.414IoU 0.828 0.796 0.919 0.861 0.894 0.360 0.711 0.286 0.405 0.346
MedSAM-Box Dice 0.946 0.912 0.976 0.903 0.955 0.692 0.918 0.856 0.896 0.846IoU 0.899 0.842 0.953 0.832 0.918 0.572 0.851 0.752 0.815 0.743
IMISNet-Point Dice 0.719 0.444 0.764 0.462 0.694 0.386 0.726 0.257 0.483 0.677IoU 0.590 0.304 0.647 0.316 0.564 0.271 0.607 0.171 0.335 0.528
IMISNet-Box Dice 0.948 0.639 0.916 0.465 0.838 0.430 0.873 0.209 0.625 0.841IoU 0.901 0.535 0.859 0.368 0.757 0.349 0.791 0.148 0.509 0.733
LISA Dice 0.243 0.621 0.220 0.487 0.187 0.177 0.218 0.615 0.321 0.036IoU 0.184 0.469 0.163 0.325 0.126 0.111 0.147 0.466 0.227 0.019
GLAMM Dice 0.192 0.504 0.178 0.473 0.173 0.190 0.315 0.611 0.282 0.027IoU 0.136 0.344 0.121 0.325 0.117 0.121 0.231 0.463 0.189 0.014
HyperSeg Dice 0.329 0.413 0.330 0.463 0.323 0.157 0.536 0.609 0.157 0.023IoU 0.239 0.266 0.246 0.305 0.227 0.103 0.450 0.458 0.100 0.012
Seg-R1 Dice 0.446 0.583 0.383 0.522 0.385 0.186 0.625 0.613 0.233 0.455IoU 0.342 0.445 0.312 0.372 0.313 0.121 0.521 0.463 0.162 0.325
MedPlib Dice 0.225 0.086 0.076 0.030 0.156 0.116 0.135 0.088 0.042 0.297IoU 0.154 0.049 0.048 0.017 0.106 0.073 0.098 0.048 0.023 0.201
UniBiomed Dice 0.936 0.887 0.959 0.871 0.894 0.357 0.828 0.620 0.761 0.794IoU 0.885 0.804 0.923 0.782 0.830 0.262 0.743 0.470 0.640 0.668
Citrus-V Dice 0.694 0.702 0.713 0.799 0.653 0.186 0.324 0.040 0.038 0.782IoU 0.601 0.600 0.623 0.716 0.542 0.136 0.237 0.021 0.022 0.655
Ours-IMISNet Dice 0.947 0.832 0.941 0.843 0.902 0.324 0.788 0.432 0.756 0.803IoU 0.899 0.729 0.892 0.747 0.829 0.242 0.699 0.294 0.641 0.678
Ours-MedSAM2 Dice 0.930 0.900 0.973 0.874 0.948 0.375 0.805 0.777 0.798 0.813IoU 0.876 0.828 0.950 0.796 0.906 0.291 0.719 0.647 0.689 0.692
22
MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
Origin Image Turn 1 IoU:0.1153 Turn 2 IoU:0.7901 Turn 3 IoU:0.9189 Groundtruth
Please segment the left heart ventricle in heart MRI
Please segment the kidney in kidney CT
Origin Image Turn 1 IoU:0.6680 Turn 2 IoU:0.8856 Turn 3 IoU:0.9315 Groundtruth
Please segment the benign tumor in breast ultrasound
Origin Image Turn 1 IoU:0.5793 Turn 2 IoU:0.7999 Turn 3 IoU:0.9004 Groundtruth
Origin Image Turn 1 IoU:0.2028 Turn 2 IoU:0.5547 Turn 3 IoU:0.9689 Groundtruth
Please segment the liver in abdomen CT
Please segment the neoplastic polyp in colon endoscope
Origin Image Turn 1 IoU:0.5837 Turn 2 IoU:0.7721 Turn 3 IoU:0.8368 GroundtruthTurn 3 IoU:0.9050
Figure 4. Case study. Yellow boxes indicate bounding box prompts, yellow points represent positive clicks, and red points denote negative
clicks.
23