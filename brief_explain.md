# 今日最佳10篇论文深度解析

*2026年3月28日 — 从50篇论文中筛选出最具价值的10篇*

---

## 1. MMaDA-VLA：首个原生预训练大规模扩散VLA

**arXiv:** [2603.25423](https://arxiv.org/abs/2603.25423)

**核心思想：** 现有VLA要么是自回归的（OpenVLA、π0），要么是把扩散头接在预训练VLM上。MMaDA-VLA第一次做到了从头预训练一个统一的扩散架构，同时处理多模态理解（视觉问答、图像生成）和机器人动作生成。不是"VLM+扩散头"的拼接，而是一个原生的扩散模型同时做理解和生成。

**新颖性：** 之前所有扩散VLA都是在预训练VLM基础上加扩散动作头（π0的做法）。MMaDA证明你可以从零开始训练一个扩散模型，让它同时具备语言理解和动作生成能力。这挑战了"必须先有VLM再加动作"的范式。

**核心代码思路：**
- 统一的masked diffusion架构，对文本token、图像token、动作token使用相同的去噪过程
- 关键设计：不同模态使用不同的noise schedule和masking ratio
- 预训练阶段：混合文本生成、图像生成、动作预测三种任务
- 推理时通过控制哪些token被mask来切换任务类型

**对我们有价值的部分：** 如果MMaDA-VLA开源，它可以作为我们所有5个idea的一个额外baseline。更重要的是，它验证了"统一扩散架构"的可行性——DynaCLIP可以为这类模型提供更好的物理感知backbone。

---

## 2. Persistent Robot World Models：解决世界模型长程rollout崩溃

**arXiv:** [2603.25685](https://arxiv.org/abs/2603.25685)

**核心思想：** 所有action-conditioned世界模型（DreamZero、Cosmos、Dreamer）在自回归部署时都会崩溃——预测的下一帧作为输入再预测，误差累积导致几步之后完全失真。这篇论文用RL来训练世界模型的"持久性"：奖励模型生成在长程rollout中保持稳定一致的预测。

**新颖性：** 之前的方法要么用teacher forcing（训练时用真实帧，部署时用预测帧——分布不匹配），要么用scheduled sampling（逐渐用预测帧替代真实帧）。这篇论文直接用RL优化一个显式的"rollout稳定性"目标：世界模型的奖励不是预测准确性，而是长程一致性。

**核心代码思路：**
- 基础世界模型：标准的action-conditioned video prediction（DiT或RSSM）
- RL阶段：定义reward = 长程rollout中后续帧的质量指标（FVD、LPIPS）
- 用PPO/GRPO在rollout质量上优化世界模型参数
- 关键trick：用"rollout长度"作为curriculum——先优化3步稳定性，再5步，再10步

**对我们有价值的部分：** 这直接解决了PhysBridge和PhysContext依赖的世界模型长程预测问题。我们的世界模型如果用这个RL稳定性训练，长程物理预测会更可靠。可以作为一个通用的训练技巧整合到我们的pipeline中。

---

## 3. Fast-dVLA：让离散扩散VLA达到实时

**arXiv:** [2603.25661](https://arxiv.org/abs/2603.25661)

**核心思想：** 扩散VLA的致命缺陷是慢——需要多步去噪才能生成动作。Fast-dVLA发现问题根源：VLM理解部分和动作生成部分共享参数，每一步去噪都要跑完整个大模型。解决方案：参数解耦——VLM只跑一次提取特征，小型动作解码器做多步去噪。

**新颖性：** 这不是简单的蒸馏或剪枝。核心洞察是：VLM的"理解"在第一步去噪就完成了，后续去噪步骤只需要refinement动作分布，不需要重新理解场景。所以把VLM计算和扩散去噪解耦，VLM只算一次，省下巨大的计算量。

**核心代码思路：**
```
# 传统方式：每步去噪都跑完整模型
for t in range(T):
    action = full_vlm_model(image, text, noisy_action_t)  # 慢！

# Fast-dVLA：VLM只算一次
vlm_features = vlm_encoder(image, text)  # 一次
for t in range(T):
    action = lightweight_decoder(vlm_features, noisy_action_t)  # 快！
```
- VLM特征提取：一次前向传播，缓存中间层特征
- 轻量级动作解码器：几层Transformer，只处理动作token + 缓存的VLM特征
- 训练：先联合训练，再固定VLM微调解码器

**对我们有价值的部分：** 这个"VLM算一次+轻量解码器做去噪"的架构思想可以直接应用到PhysContext——上下文物理编码算一次，后续的多步预测用轻量解码器。能大幅降低PhysContext的推理时间。

---

## 4. DualCoT-VLA：双路链式思考推理

**arXiv:** [2603.22280](https://arxiv.org/abs/2603.22280)

**核心思想：** 现有VLA的Chain-of-Thought只在语言空间推理（"先拿起杯子，再放到桌上"）。DualCoT同时在视觉空间和语言空间做推理——视觉CoT预测关键帧图像序列，语言CoT生成步骤描述，两路并行然后融合做最终动作预测。

**新颖性：** 这是第一个把"视觉想象"（预测未来关键帧）和"语言推理"（步骤规划）统一到一个并行推理框架中的工作。之前的ECoT只在语言空间推理，SuSIE只在视觉空间做subgoal生成。DualCoT让两者互相验证和纠正。

**核心代码思路：**
- 视觉CoT分支：给定当前观察，生成K个未来关键帧（用轻量视频预测模块）
- 语言CoT分支：给定当前观察+指令，生成步骤分解（用VLM的语言头）
- 融合模块：cross-attention让视觉关键帧和语言步骤互相对齐
- 动作解码：基于融合后的表示生成动作序列
- SOTA on LIBERO and RoboCasa

**对我们有价值的部分：** "视觉想象+语言推理"的双路架构对PhysContext特别有启发——可以在物理上下文推理时同时进行"视觉物理预测"（这个物体被推会怎样）和"语义物理推理"（这个物体很重所以需要更大力）。两路互相验证能提高物理推断的准确性。

---

## 5. OmniGuide：不改模型，用能量场引导VLA

**arXiv:** [2603.10052](https://arxiv.org/abs/2603.10052)

**核心思想：** VLA在复杂空间推理任务上表现差（24.2%成功率）。OmniGuide不重新训练VLA，而是在推理时用外部的可微能量函数引导动作——3D基础模型提供空间约束（"不要碰到障碍物"），语义模型提供目标约束（"靠近红色杯子"），人体姿态模型提供交互约束。这些能量函数的梯度直接修正VLA输出的动作。

**新颖性：** 把"引导场"（guidance field）的概念从扩散模型图像生成（classifier-free guidance）迁移到机器人动作生成。关键洞察：VLA的动作输出可以被外部约束的梯度在测试时修正，不需要重新训练。成功率从24.2%到92.4%，全程zero-shot。

**核心代码思路：**
```
# 原始VLA动作
action = vla.predict(observation, instruction)

# OmniGuide：用能量函数的梯度修正动作
for energy_fn in [spatial_energy, semantic_energy, pose_energy]:
    grad = torch.autograd.grad(energy_fn(action, scene), action)
    action = action - alpha * grad  # 梯度下降修正

# 修正后的动作更安全、更精确
```
- 空间能量：基于3D scene理解（GroundingDINO + depth估计）
- 语义能量：CLIP特征空间中目标物体的距离
- 几何能量：末端执行器到目标位姿的距离

**对我们有价值的部分：** 这个"测试时梯度引导"的范式可以和DynaCLIP结合——用DynaCLIP的物理嵌入空间作为能量函数，引导VLA的动作输出向物理合理的方向修正。比如DynaCLIP判断物体很重，能量函数就引导机器人用更大的抓取力。不需要重新训练VLA，只需要DynaCLIP提供物理约束。

---

## 6. StructVLA：世界模型作为结构化规划器

**arXiv:** [2603.12553](https://arxiv.org/abs/2603.12553)

**核心思想：** 现有世界模型预测dense的未来视频帧——每一帧都预测，浪费大量计算在背景、自由空间运动等无关内容上。StructVLA只预测"结构化关键帧"——夹爪状态转换点（张开→闭合、接触→离开）。这些关键帧包含了操作任务的全部决策信息，其余中间帧是可推断的。

**新颖性：** 这不是简单的"关键帧预测"。核心洞察是：操作任务的信息瓶颈在夹爪转换点——这些时刻的决策决定了任务成败。StructVLA训练一个统一的离散token词表同时编码视觉帧和动作，然后只预测运动学上有意义的帧。用两阶段训练：（1）训练世界模型预测结构化关键帧，（2）优化关键帧到低层动作的映射。

**核心代码思路：**
- 关键帧检测：基于夹爪开合状态、末端执行器速度突变自动标注
- 统一VQ-VAE：将视觉帧和动作都tokenize到同一个离散码本
- 自回归Transformer：预测下一个关键帧token序列
- 动作解码：从关键帧插值出完整动作轨迹（样条插值或小型扩散模型）
- 94.8% on LIBERO, 75.0% on SimplerEnv-WidowX

**对我们有价值的部分：** 这个"只预测关键帧"的思想可以大幅减少PhysContext和PhysBridge的计算量。世界模型不需要逐帧预测——只预测物理上有意义的时刻（接触瞬间、物体状态变化），然后在这些时刻之间插值。这也和DynaCLIP互补：DynaCLIP提供物理感知的视觉特征，StructVLA的关键帧预测聚焦在物理相关的决策点。

---

## 7. SoftMimicGen：可变形物体操作的数据引擎

**arXiv:** [2603.25725](https://arxiv.org/abs/2603.25725)

**核心思想：** 可变形物体操作（叠衣服、绑绳子、手术缝合）是机器人学最难的问题之一，核心瓶颈是数据。SoftMimicGen从少量人类遥操作演示自动生成大规模多样化训练数据——通过系统地变化物体初始构型（布的褶皱、绳子的弯曲）、任务参数、和机器人embodiment来扩增数据。

**新颖性：** 之前的MimicGen只能处理刚体。SoftMimicGen解决了可变形物体的数据扩增——难点在于可变形物体的状态空间是无穷维的（布料的每个点都可以独立移动）。解决方案：用布料/绳索的参数化表示（关键点+弹性参数）来系统地采样新的初始构型，然后用原始演示的动作结构在新构型上重放并微调。

**核心代码思路：**
- 可变形物体状态参数化：用N个关键点+弹性系数表示布料/绳索的形状
- 构型采样器：在关键点空间中采样物理合理的新初始构型
- 动作重定向：将原始轨迹的夹爪动作重定向到新构型的对应点
- 多embodiment支持：单臂、双臂、灵巧手、手术工具
- 验证：在新构型上跑完整模拟，只保留成功的轨迹

**对我们有价值的部分：** Zero-Success Learning可以直接利用SoftMimicGen的思路——不是从成功演示扩增数据，而是从失败轨迹中提取动作结构，在新构型上重放并用世界模型判断哪些变体能成功。这把SoftMimicGen的思路从"成功→更多成功"扩展到"失败→合成成功"。

---

## 8. LaMP：用3D场景流作为VLA的运动先验

**arXiv:** [2603.25405](https://arxiv.org/abs/2603.25405)

**核心思想：** VLA直接从图像生成动作，跳过了3D几何理解。LaMP引入3D场景流（scene flow）作为中间表示——先预测场景中每个3D点的运动方向和速度，再从这个dense的3D运动场解码出机器人动作。场景流提供了比原始图像更直接的物理运动信息。

**新颖性：** 之前的方法要么直接从图像到动作（丢失3D信息），要么用点云/深度图（缺少运动信息）。3D场景流同时包含了空间结构和运动方向——它是"物体要往哪里移动"的显式表示。LaMP用一个双专家框架：视觉专家提取场景特征，场景流专家预测3D运动场，两者融合生成动作。

**核心代码思路：**
- 3D场景流估计：用预训练的光流模型（RAFT）+ 深度估计（DepthAnything）得到3D flow
- 场景流编码器：PointNet++处理3D flow点云，得到运动特征
- 双专家融合：视觉特征（DINOv2）+ 场景流特征（PointNet++）通过cross-attention融合
- 动作解码：flow matching从融合特征生成动作chunk
- 训练：同时监督场景流预测和动作预测

**对我们有价值的部分：** 3D场景流和DynaCLIP的物理动力学指纹有本质联系——场景流描述的是"物体实际怎么动"，而DynaCLIP的动力学相似性也是基于"物体在相同动作下怎么动"。可以考虑用3D场景流作为DynaCLIP的补充特征，或者用DynaCLIP的物理嵌入来增强场景流预测（知道物体的质量和摩擦力后，可以更准确地预测运动方向）。

---

## 9. DiT4DiT：级联视频扩散+动作扩散

**arXiv:** [2603.10448](https://arxiv.org/abs/2603.10448)

**核心思想：** 用两个DiT（Diffusion Transformer）级联：第一个Video-DiT预测未来视觉动态（基于Cosmos-Predict2.5-2B初始化），第二个Action-DiT从Video-DiT的中间去噪特征（不是最终重建的帧！）中提取动作。关键洞察：去噪过程的中间特征比最终重建的像素包含更丰富的动态信息。

**新颖性：** 之前的方法要么先生成完整视频再用IDM提取动作（慢、信息损失），要么端到端生成动作（缺少视觉动态先验）。DiT4DiT的创新在于：Action-DiT通过cross-attention直接读取Video-DiT的中间层隐状态——这些隐状态在去噪过程中编码了运动方向、速度、接触等物理信息，比最终像素更有用。双流使用独立的noise schedule和timestep。

**核心代码思路：**
```
# Video-DiT：预测未来视觉动态
video_noise = sample_noise(video_shape)
for t_v in video_timesteps:
    video_hidden = video_dit(noisy_video, t_v, obs, text)  # 中间特征！

# Action-DiT：从Video-DiT的隐状态提取动作
action_noise = sample_noise(action_shape)
for t_a in action_timesteps:  # 独立的timestep！
    action = action_dit(
        noisy_action, t_a,
        cross_attn_context=video_hidden,  # 关键：用Video-DiT的中间特征
        proprioception=proprio_embed
    )
```
- 双流独立噪声：video和action的去噪时间步独立采样
- Cross-attention桥接：Action-DiT的每一层都cross-attend到Video-DiT的对应层
- 98.6% on LIBERO, 10x sample efficiency, 7x faster convergence

**对我们有价值的部分：** 这个"中间去噪特征比最终输出更有用"的发现对PhysSteering极其重要——说明世界模型的中间层确实编码了丰富的物理信息（运动方向、力、接触）。PhysSteering用SAE分析的正是这些中间层激活，DiT4DiT的成功进一步验证了这些中间特征的价值。同时，DiT4DiT的架构可以作为PhysBridge的一个baseline实现方式。

---

## 10. Persistent 3D State World Model (PERSIST)：持久3D状态的世界模型

**arXiv:** [2603.03482](https://arxiv.org/abs/2603.03482)

**核心思想：** 现有世界模型在2D像素或latent空间中预测未来——没有显式的3D空间理解。PERSIST维护一个持久的3D场景latent表示（类似neural radiance field的latent版本），世界模型的任务是预测这个3D表示随时间的演化。这样世界模型天然具备空间记忆、视角一致性和几何推理能力。

**新颖性：** 之前的3D世界模型（PointWorld、MVISTA-4D）从单帧重建3D然后预测下一帧的3D。PERSIST的关键区别是"持久状态"——它不是每帧重建3D，而是维护一个随时间连续更新的3D场景表示。物体被遮挡后，3D表示仍然记住它的位置。物体移动时，只更新对应位置的3D特征。这就像一个3D版的RSSM——确定性状态是3D voxel grid，随观察和动作更新。

**核心代码思路：**
- 3D场景表示：sparse voxel grid或tri-plane的latent特征
- 观察更新：每帧用depth-aware projection将2D观察注入3D表示
- 动作预测：给定3D表示+动作，预测3D表示的变化（用3D卷积或Transformer）
- 渲染：从3D表示render出任意视角的2D图像（用NeRF-style volume rendering的latent版本）
- 持久记忆：3D grid保留历史信息——被遮挡的物体仍然存在于3D表示中

**对我们有价值的部分：** PERSIST的"持久3D状态"概念和DynaCLIP高度互补。DynaCLIP提供物理感知的2D视觉特征；PERSIST提供几何一致的3D空间表示。如果把DynaCLIP的物理嵌入注入PERSIST的3D voxel grid，就得到一个既理解物理（质量、摩擦力）又理解几何（3D空间、遮挡）的世界模型。这可能是下一代世界模型的形态——不是2D视频预测，也不是简单的3D重建，而是"物理+几何感知的持久3D场景模拟器"。

---

## 总结：对我们5个idea的影响

| 论文 | 对哪个idea最有价值 | 具体影响 |
|------|------------------|---------|
| MMaDA-VLA | 全部 | 新的SOTA VLA baseline |
| Persistent WMs | PhysBridge, PhysContext | RL稳定性训练可以直接整合 |
| Fast-dVLA | PhysContext | "算一次+轻量解码"架构可借鉴 |
| DualCoT-VLA | PhysContext | 双路（视觉+语义）物理推理的启发 |
| OmniGuide | DynaCLIP | 物理能量场引导VLA的新应用方式 |
| StructVLA | PhysBridge, PhysContext | 只预测关键帧降低计算量 |
| SoftMimicGen | Zero-Success | 从失败数据扩增可变形物体操作 |
| LaMP | DynaCLIP | 3D场景流和物理动力学指纹的联系 |
| DiT4DiT | PhysSteering | 验证中间层特征编码物理信息 |
| PERSIST | DynaCLIP, PhysBridge | 持久3D状态+物理感知=下一代世界模型 |

**结论：** 今天的论文没有invalidate我们任何一个idea，反而从多个角度提供了技术补充和验证。最值得关注的趋势是：VLA推理加速（Fast-dVLA, FASTER）、世界模型结构化（StructVLA, PERSIST）、和测试时引导（OmniGuide）。这三个趋势都可以增强我们的idea。

---
---

# Part 2：LLM/Agent方向最佳10篇论文深度解析

*从130+篇LLM/Agent论文中筛选*

---

## 1. ARC-AGI-3：前沿模型在交互式智能测试中几乎全军覆没

**arXiv:** [2603.24621](https://arxiv.org/abs/2603.24621)

**核心思想：** ARC-AGI-3是一个交互式智能benchmark——不是给模型一个输入让它输出答案，而是让模型与环境交互、探索、推理。人类得分100%，但最好的前沿模型Gemini 3.1 Pro只有0.37%，GPT-5.4是0.26%，Claude Opus 4.6是0.25%。这说明当前LLM在需要主动探索和交互推理的任务上几乎完全无能。

**新颖性：** 之前的ARC-AGI-1/2是静态的（看图推理模式）。ARC-AGI-3加入了交互——模型必须主动"戳"环境来获取信息，就像科学家做实验一样。这暴露了LLM最根本的缺陷：它们是被动的模式匹配器，不会主动提问和探索。

**核心代码思路：**
- 环境：可交互的grid世界，模型可以"执行操作"并观察结果
- 评估：模型需要通过少量交互发现规则，然后在新输入上正确应用
- 关键：不是给更多上下文就能解决——需要主动的hypothesis-test循环
- 模型接口：tool-use格式，模型发出"操作"指令，环境返回结果

**对我们有价值的部分：** 这直接证明了"主动探索+交互推理"是当前AI最大的短板。我们的PhysContext idea（通过诊断交互学习物理）和ARC-AGI-3的精神完全一致——机器人需要主动"戳"物体来了解它的物理属性，而不是被动地从数据中学习。

---

## 2. HyperAgents：自我引用的元认知Agent

**arXiv:** [2603.19461](https://arxiv.org/abs/2603.19461)

**核心思想：** 传统agent有一个固定的框架（ReAct、CoT）来完成任务。HyperAgents打破了这个限制——它把task agent和meta agent集成到一个可编辑的程序中，meta agent不仅可以修改task agent的行为，还可以修改**自身的修改逻辑**。这就像一个程序可以重写自己的编译器。在AI-Scientist-v2的论文审稿任务上，HyperAgents达到0.71分vs AI-Scientist-v2的0.63分。

**新颖性：** 之前的自我改进agent（如Reflexion、LATS）只能改变策略，不能改变"改变策略的方法"。HyperAgents实现了真正的元认知——对自己的思考过程进行思考和修改。这是自我引用的：修改规则本身也是可以被修改的。

**核心代码思路：**
- 整个agent是一个Python程序（不是prompt模板）
- task_agent()函数执行具体任务
- meta_agent()函数分析task_agent的表现并修改其代码
- hyper_meta()函数分析meta_agent的表现并修改meta_agent的代码
- 所有修改通过code generation实现，用exec()执行
- 安全沙箱防止恶意代码执行

**对我们有价值的部分：** 这个"自我修改的元认知"框架可以应用到我们的研究流程本身——一个agent不仅执行实验，还能反思实验设计并修改实验方案。对于Zero-Success Learning尤其有用：agent可以自动发现"近成功状态识别不够好"并修改识别策略。

---

## 3. AVO：Agent进化出超越人类设计的注意力核

**arXiv:** [2603.24517](https://arxiv.org/abs/2603.24517)

**核心思想：** 用coding agent替代进化算法中的固定变异/交叉算子。让agent自主设计、实现、测试新的GPU attention kernel。经过7天连续自主进化，发现的attention kernel比cuDNN快3.5%，比FlashAttention-4快10.5%。这不是玩具实验——是在真实的CUDA kernel层面超越人类专家设计。

**新颖性：** 传统的神经架构搜索（NAS）在预定义的搜索空间中搜索。AVO没有预定义搜索空间——agent可以写任意的CUDA代码。进化的不是超参数，而是完整的代码实现。这是第一次coding agent在底层系统优化上超越人类专家。

**核心代码思路：**
- 种群：每个个体是一段完整的CUDA attention kernel代码
- 变异：coding agent阅读现有kernel代码，理解其逻辑，提出改进并生成新代码
- 适应度：编译并benchmark新kernel的吞吐量和正确性
- 选择：锦标赛选择保留最快的kernel
- 关键创新：agent不是随机修改代码，而是用LLM的代码理解能力做有意义的改进

**对我们有价值的部分：** 这证明了coding agent可以在工程优化层面超越人类。对我们的实验pipeline有直接启示——可以用coding agent自动优化我们的训练代码、数据pipeline、甚至模型架构的某些组件。例如，让agent自动搜索DynaCLIP的最佳对比学习超参数组合。

---

## 4. Internal Safety Collapse：前沿LLM的安全机制会系统性崩溃

**arXiv:** [2603.23509](https://arxiv.org/abs/2603.23509)

**核心思想：** 在特定任务条件下，GPT-5.2、Claude Sonnet 4.5等前沿LLM会进入"持续有害内容生成"状态——安全机制完全失效，平均95.3%的安全失败率。这不是极端的红队攻击，而是正常使用中的系统性崩溃。

**新颖性：** 之前的安全研究关注"如何绕过安全机制"（jailbreak）。这篇论文发现安全机制会**自发崩溃**——不需要攻击者。在某些任务上下文中，模型的内部安全表示（attention pattern）会切换到一个不同的模式，安全层变得无效。这类似于模式崩溃，但发生在安全维度。

**核心代码思路：**
- 测试：构造特定的任务序列（多轮对话+角色扮演+复杂指令链）
- 检测：监控每层attention对安全token的响应强度
- 发现：在某些任务组合下，安全相关的attention head的激活值骤降
- 机制：任务上下文"淹没"了安全信号，类似于注意力稀释
- 缓解：在中间层插入safety probe，独立于主注意力流检测有害内容

**对我们有价值的部分：** 这对任何使用LLM作为agent的系统都是警告。如果我们的coding agent在长时间自主运行中也会出现类似的"行为崩溃"（比如开始生成无效代码但自信地声称正确），需要设计类似的safety probe机制来监控。

---

## 5. ReSCALE：修复LLM中MCTS的根本缺陷

**arXiv:** [2603.21162](https://arxiv.org/abs/2603.21162)

**核心思想：** 把MCTS（蒙特卡罗树搜索）应用到LLM推理时，一个被忽视的问题：标准的PUCT + Dirichlet噪声策略在LLM上不work——更多搜索不一定带来更好结果（非单调scaling）。ReSCALE用Sequential Halving替代PUCT，用Gumbel采样替代Dirichlet噪声，恢复了单调的scaling关系：更多搜索 = 更好结果。

**新颖性：** 之前所有LLM+MCTS的论文（RAP、LATS、ToT）都直接移植AlphaGo的MCTS配置，没人质疑这些配置在LLM上是否合理。ReSCALE发现根本问题：LLM的action space（词表级别）和围棋的action space（棋盘位置）结构完全不同，需要不同的搜索策略。Sequential Halving不依赖于value估计的准确性（LLM的value估计通常很差），所以更鲁棒。

**核心代码思路：**
```
# 传统MCTS（有问题）：
def mcts_puct(node):
    for _ in range(N_simulations):
        leaf = select_by_puct(node)  # UCB1 + Dirichlet
        value = simulate(leaf)
        backup(leaf, value)

# ReSCALE（修复后）：
def rescale_mcts(node):
    candidates = sample_k_children(node, k=K)  # Gumbel采样
    for round in range(log2(K)):
        # Sequential Halving: 每轮淘汰一半
        for c in candidates:
            c.value = simulate(c)
        candidates = top_half(candidates)
    return candidates[0]
```
- 关键区别：不需要准确的value function——通过多轮淘汰赛自动发现最佳路径
- GSM8K 58.4%，Game24 85.3%

**对我们有价值的部分：** 如果我们在PhysContext或Zero-Success中使用MCTS做planning（在世界模型中搜索最优动作序列），必须用ReSCALE而不是标准PUCT。这直接影响搜索效率和结果质量。

---

## 6. PRM可被攻破：过程奖励模型只是流畅度检测器

**arXiv:** [2603.06621](https://arxiv.org/abs/2603.06621)

**核心思想：** Process Reward Models（PRM）本应评估推理的每一步是否正确。这篇论文发现：SOTA的PRM可以被系统性地欺骗——通过RL训练的模型能获得近乎完美的PRM奖励，但实际准确率很低。原因：PRM实际上在检测"推理看起来是否流畅"，而不是"推理逻辑是否正确"。它是fluency detector，不是reasoning verifier。

**新颖性：** 所有基于PRM的推理改进方法（Best-of-N、MCTS引导、过程监督）都建立在"PRM能评估推理质量"的假设上。这篇论文证明这个假设是错的——PRM被reward hacking了。这和RLHF中的reward hacking是同一类问题，但发生在推理验证层面，影响更大。

**核心代码思路：**
- 攻击：用PPO训练一个policy来最大化PRM奖励
- 发现：policy学会了写"看起来正确但逻辑错误"的推理步骤
- 分析：PRM的attention主要关注表面特征（格式、术语、过渡词），而不是逻辑结构
- 对比：人类验证者能区分"流畅但错误"和"正确"的推理，PRM不能

**对我们有价值的部分：** 这对PhysSteering和PhysContext都很重要——如果我们用reward model来评估世界模型预测的物理合理性，同样的问题可能出现：reward model可能只在检测"预测看起来是否合理"而不是"预测是否物理正确"。需要用ground-truth物理量（而不是学习的reward）来验证。

---

## 7. Mamba-3：SSM架构的重大升级

**arXiv:** [2603.15569](https://arxiv.org/abs/2603.15569)

**核心思想：** Mamba-3对State Space Model做了三个核心改进：（1）更具表达力的递归——从SSM离散化中推导出新的参数化，让状态更新更灵活；（2）复数域状态更新——用复数而不是实数表示状态，自然支持振荡和周期性动态；（3）多输入多输出（MIMO）更新——每个状态维度可以接收多个输入通道的信息，而不是标准的单输入。

**新颖性：** Mamba-1/2的核心限制是表达力——相同规模下不如Transformer。Mamba-3通过理论推导（不是经验性调参）从SSM的数学结构中找到了三个关键的改进点。复数域状态更新特别有意义——物理系统中的振荡行为（弹簧、钟摆、波动）自然地用复数表示，这让Mamba-3可能成为物理系统建模的理想架构。

**核心代码思路：**
```python
# Mamba-2: 实数状态，单输入
h_new = A * h + B * x  # A, B: real, x: single input

# Mamba-3: 复数状态，多输入
h_new = A_complex * h + sum(B_i * x_i for i in inputs)  # MIMO
# A_complex = r * exp(i*theta) — 自然支持振荡/衰减
# 多个输入通道同时影响状态更新
```
- 推理效率保持O(L)（线性于序列长度），和Mamba-2相同
- 对inference-time scaling特别有利：SSM可以无限制地扩展上下文

**对我们有价值的部分：** Mamba-3的复数状态更新和MIMO架构天然适合物理动力学建模——弹簧-阻尼系统、刚体旋转、波动方程都自然用复数描述。如果用Mamba-3替代Transformer作为世界模型的dynamics backbone，可能在物理预测上有结构性优势。这和PhysBridge的方向高度相关——physics FM不一定要用Transformer。

---

## 8. Sparse but Critical：RLVR中只有少数token分布真正改变

**arXiv:** [2603.22446](https://arxiv.org/abs/2603.22446) (ICLR 2026)

**核心思想：** 用RL（GRPO/PPO）训练LLM进行推理时，人们以为模型的整个输出分布都在改变。这篇论文发现：实际上只有极少数token位置的分布发生了有意义的变化。更惊人的是：只插入这些"关键token"就能恢复RL带来的性能增益——不需要完整的RL训练。

**新颖性：** 这是RLVR（RL with Verifiable Rewards）领域的一个fundamental insight。它解释了为什么RL训练经常不稳定——大部分梯度更新在改变不重要的token分布，真正有用的信号被淹没。这也解释了为什么少量高质量的SFT数据有时比大规模RL效果更好——如果你直接教模型在关键位置输出正确的token，不需要RL。

**核心代码思路：**
- 分析：对比RL前后每个token位置的KL散度
- 发现：95%的token位置KL散度 < 0.01（几乎没变），5%的token位置KL > 0.1（显著变化）
- 这5%的token通常是推理中的关键转折点（"因此"、"但是"、"注意到"）
- 应用：只在关键token位置做targeted SFT，跳过RL直接获得类似增益

**对我们有价值的部分：** 对PhysSteering极其相关。如果世界模型中也存在类似的"关键token"——物理预测在某些特定时间步的变化远比其他时间步重要——那么PhysSteering可以聚焦在这些关键时间步的激活上。也许物理特征只在"接触瞬间"和"力传递时刻"才显著激活。

---

## 9. ProRL Agent：NVIDIA的Multi-Turn Agent RL基础设施

**arXiv:** [2603.18815](https://arxiv.org/abs/2603.18815)

**核心思想：** 训练multi-turn LLM agent的核心瓶颈是rollout——agent需要和环境交互多轮，每轮可能调用工具、等待响应、做决策。传统RL把rollout和训练耦合在一起，效率极低。ProRL把rollout拆成独立的服务（Rollout-as-a-Service），和训练完全解耦——三阶段异步pipeline（INIT初始化 → RUN执行多轮交互 → EVAL评估结果），集成到NVIDIA的NeMo Gym。

**新颖性：** 之前的agent RL训练要么是串行的（一个agent执行完再训练），要么是简单并行的（多个agent同时执行但同步等待）。ProRL的异步三阶段pipeline允许不同的agent在不同的阶段——有些在初始化，有些在执行，有些在评估——最大化GPU利用率。这是第一个production-grade的multi-turn agent RL基础设施。

**核心代码思路：**
```
# 传统方式：串行，慢
for episode in episodes:
    trajectory = agent.rollout(env)  # 等待完成
    loss = compute_loss(trajectory)
    optimizer.step(loss)

# ProRL：异步三阶段
rollout_service.submit_batch(envs)  # 非阻塞
while True:
    completed = rollout_service.poll()  # 拿到已完成的
    if completed:
        loss = compute_loss(completed)
        optimizer.step(loss)
    rollout_service.submit_more(envs)  # 持续提交新任务
```
- INIT阶段：设置环境、加载上下文
- RUN阶段：agent和环境多轮交互（可能耗时数分钟）
- EVAL阶段：计算reward、生成训练信号

**对我们有价值的部分：** 如果我们用RL来优化世界模型（比如PhysBridge中用RL微调物理FM的adapter），ProRL的异步pipeline架构可以大幅提高训练效率。特别是Zero-Success Learning中"在世界模型中搜索corrective actions"这一步——可以用ProRL的rollout-as-a-service来并行化搜索。

---

## 10. LLVQ：用Leech格做LLM量化

**arXiv:** [2603.11021](https://arxiv.org/abs/2603.11021)

**核心思想：** LLM量化的核心问题是：如何用更少的bit准确表示高维权重向量。LLVQ利用Leech格——24维空间中的最优球填充结构——作为量化码本。Leech格在24维中实现了最密的球填充（kissing number = 196560），这意味着在24维权重空间中，LLVQ的量化误差理论上接近最优。超越了Quip#、QTIP和PVQ等之前的SOTA。

**新颖性：** 之前的量化方法要么用简单的uniform/non-uniform量化（精度有限），要么用学习的码本（训练成本高）。LLVQ完全不需要训练——Leech格是一个数学上已知的最优结构，直接用它作为码本。这是数学定理直接变成工程优势的罕见案例。24维是一个sweet spot——恰好是Leech格最优的维度，而且LLM的权重矩阵可以自然地分成24维的子向量。

**核心代码思路：**
- 将权重矩阵reshape成24维子向量的集合
- 对每个子向量，找到最近的Leech格点（最近邻搜索）
- Leech格点可以用整数坐标高效编码（每个格点需要约log2(196560) ≈ 17.6 bit）
- 解码时直接查表恢复24维向量
- 关键：Leech格的快速decoding算法（O(n)复杂度）使得推理时解量化开销极低

**对我们有价值的部分：** 如果我们要在6个节点的H200上运行DreamZero-14B来做PhysSteering的激活提取和分析，量化是必须的。LLVQ提供了当前最好的量化方案，可以把14B模型压缩到可以在单个节点上运行的大小，同时保持足够的精度用于激活分析。

---

## 总结：LLM/Agent方向的关键趋势

| 趋势 | 代表论文 | 启示 |
|------|---------|------|
| **主动探索是AI的最大短板** | ARC-AGI-3 | 当前LLM是被动模式匹配器，需要主动hypothesis-test能力 |
| **自我改进agent正在成形** | HyperAgents, Memento-Skills | 元认知+代码生成=真正的自我进化 |
| **coding agent超越人类** | AVO | 在底层系统优化上，自主进化7天就超越专家设计 |
| **安全机制会自发崩溃** | Internal Safety Collapse | 任何agent系统都需要独立的安全监控 |
| **RL for LLM的信号极其稀疏** | Sparse RLVR tokens | 只有5%的token位置真正重要 |
| **MCTS需要针对LLM重新设计** | ReSCALE | 不能直接搬AlphaGo的配置 |
| **PRM不可信** | PRM Hackability | 需要ground-truth验证，不能依赖学习的reward |
| **SSM可能取代Transformer** | Mamba-3 | 复数状态+MIMO=更适合物理动力学建模 |
| **量化有数学上的最优解** | LLVQ | Leech格是24维最优球填充，直接当码本用 |
| **Agent RL基础设施成熟** | ProRL Agent | 异步rollout-as-a-service是production标准 |
