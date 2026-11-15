# CoFL-S 变形无人机项目开发上下文

整理日期：2026-11-10。用于在手机或新的 ChatGPT 对话中继续讨论。本文件是讨论摘要，不是已完成实现的报告。

## 1. 用户目标与当前决定

- 基于当前 CoFL 仓库开发，主要复用 **CoFL-S**，不使用原始 BEV CoFL 作为主模型。
- 面向**变形无人机**：关注机体上多个点的运动，这些点需要联合满足几何、运动等约束，并优化用户定义的目标函数。具体约束方程、目标函数及状态表示尚未确定。
- 曾讨论固定高度飞行、在虚拟水平平面上输出二维流场；最终是否只在二维平面运动，尚未最终确定。
- 初期考虑仅深度输入、普通 CNN 或 DINO；后续更新为计划采用 **DFormer RGB-D encoder**。
- 不需要语言指令输入（原模型是文本指令，不是语音输入）。计划使用内置目标坐标条件。
- 希望继续使用现有 CoFL-S 流场 decoder 的结构和输出形式。但多点耦合运动是否适合原输出形式，仍需论证。
- 这是独立发展分支，今后不打算合并回原 CoFL。建议自己的远程仓库作为 origin；尚未执行新建仓库或修改 remote。
- 当前没有开始 DFormer 替换，也没有实现无人机约束模块。

## 2. 涉及的仓库与资料

### 当前 CoFL 源码仓库

- 本地目录：`/home/kuma/ros/learning_ws/CoFL`
- 同时包含两项工作：
  - CoFL：鸟瞰 RGB＋语言，查询图像平面坐标，输出二维图像向量。
  - CoFL-S：第一人称 RGB-D＋语言，查询局部水平扇区坐标，输出机器人坐标系 `(forward, left)` 二维向量。
- 本次没有核实当前 Git remote 的 URL，后续应执行 `git remote -v`，不要猜测上游 GitHub 地址。
- 模型说明：`docs/models.md`
- 数据生成：`docs/generation.md`
- 推理规范：`docs/inference.md`
- 数据格式：`docs/dataset-format.md`
- 训练说明：`docs/training.md`
- 学习背景：`transformer_context.md`。后续回答沿用其中 PDF 编号、数学符号、每 token 一行的约定，不从基础重复讲起。

### DFormer 官方仓库

- https://github.com/VCIP-RGBD/DFormer
- 包含 DFormer、DFormerv2、DFormer++；具体采用哪个版本尚未最终决定。
- 官方 encoder 文件：
  - `models/encoders/DFormer.py`
  - `models/encoders/DFormerv2.py`
- 目标是复用 RGB-D encoder，**不使用其语义分割 decoder 和整套分割训练框架**。
- 原版依赖 mmcv、mmengine；与本项目环境的实际兼容性尚未运行验证。
- 上游根 LICENSE 为 MIT，但 README 另有 non-commercial 表述。保留来源与许可，发布前需确认适用条件。

### CoFL-S 发布资产

- 模型：https://huggingface.co/lhk66666/CoFL-S
- 数据：https://huggingface.co/datasets/lhk66666/CoFL-S-Dataset
- 已通过 Hugging Face API 查看模型：`best.ckpt` 为 **5,336,379,647 字节，约 5.34 GB / 4.97 GiB**。
- 完整 checkpoint 包含模型及恢复相关内容；文件大小不等于纯参数大小或运行显存。
- 已查到数据仓库有 92 个归档，总计约 **329.48 GiB**。
- 下载文件需要 HF 授权与登录。当时本机未配置 HF token，实际文件请求返回访问受限。
- 当时当前磁盘可用约 318 GiB，无法同时存下全量归档，解包另需空间。
- 没有下载该训练数据集。

## 3. 当前 CoFL-S 模型结构

```text
RGB → SigLIP 视觉塔 → RGB patch tokens ─→ 深度 FiLM 调制 ─┐
Depth + valid mask → depth patch adapter ────────────────┘
                                                        ↓
                                                  视觉投影 + LN
                                                        ↓
Text → SigLIP 文本塔 → pooled token + text tokens → 文本投影 + LN
                                                        ↓
                                         4 层视觉语言融合 → LN
                                                        ↓
                                              context C [B,Nc,d]
                                                ├→ 流场 decoder
                                                └→ Action Head
```

正式配置 `configs/cofl_s_formal.yaml`：

- `d_model=768`，任务网络 `num_heads=8`，每 head 为 96 维。
- 4 层视觉语言融合，2 层流场 decoder。
- SigLIP 为 `google/siglip2-base-patch16-224`。
- 224×224 图像得到 196 个视觉 patch token。
- 文本默认 64 个位置，外加 pooled prefix，共 65 个特征 token。
- context 仍为视觉 token 序列，不是直接拼接全部视觉和文本 token。
- 视觉与文本塔在正式配置中均 `unfreeze_last_n_layers=-1`，参与训练。
- 当前只接收单帧，没有显式时序记忆模块。

### Encoder 与 decoder 的解耦接口

```python
context = encoder(...)                     # [B,Nc,d]
query_context = decoder.prepare_context(context)
field = decoder.decode(query, query_context) # query [B,N,2] → field [B,N,2]
```

Decoder 不要求 context 一定来自 RGB、语言或 SigLIP，也不要求 Nc 必须是 196。
但现有 policy、训练数据处理、优化器分组和 checkpoint 对 SigLIP 有工程耦合，需同步修改。

### 流场 decoder

- 查询：归一化极坐标 `(theta_tilde,r_tilde)`。
- 追加 HFOV 条件，Gaussian Fourier encoding → Linear → LN。
- 经过 Pre-LN Cross-Attention＋MLP＋残差。
- Q 来自空间查询特征，K/V 来自 context。
- 当前 `need_self_attn=False`：空间 query 之间没有直接相互注意力。
- 输出方向头和非负幅值头，二者相乘得到 `(forward,left)`。
- 场向量经过尺度归一化，不是可直接发送给执行器的速度命令。

## 4. models 中脚本的职责

| 文件 | 职责 | 新项目倾向 |
|---|---|---|
| `backbone.py` | SigLIP 视觉、文本塔与 processor | 换为 DFormer 包装层 |
| `depth.py` | 深度特征对 RGB token 做 FiLM 调制 | DFormer 已融合 RGB-D，可移除旧路径 |
| `projection.py` | 视觉和文本分别投影到任务宽度 | 改为空间 token adapter |
| `fusion.py` | 视觉 Self-Attention＋读取语言的 Cross-Attention | 无语言后移除或改造 |
| `encoders.py` | 编码器总装配 | 改为 RGB-D＋目标条件 |
| `cofl_s.py` | CoFL-S 流场查询 decoder | 优先保留 |
| `blocks.py` | Attention、Fourier、Pre/Post-LN block | 保留所需部分 |
| `action.py` | 四分类动作 head | 新任务初期倾向去掉 |
| `policies.py` | encoder、decoder、action 的总接口 | 简化为新 policy |
| `cofl.py` | BEV CoFL decoder | 不作为新项目主路径 |
| `__init__.py` | 类导出 | 按新接口调整 |

## 5. Action Head 与实际部署

Action Head 和流场 decoder 是**并列分支**，都读取 context。Action Head 不是 decoder 的提前输出。

- 动作顺序：STOP、MOVE_FORWARD、TURN_LEFT、TURN_RIGHT。
- 原在线正常控制：预测流场 → 积分轨迹 → Pure Pursuit → 线速度/角速度。
- 动作 head 用于 STOP 仲裁、流场轨迹退化时的备用动作。
- 另有振荡恢复逻辑，具体优先级见 `online/dynamics.py::ActionArbitrator`。
- 因此不能把原系统简单说成“动作分类直接决定每一步方向”。
- 原 `CoFLSPolicy.forward()` 只返回流场，动作通过 `action_logits(context)` 单独调用。
- 新项目若目标坐标和位姿已知，可按距离做到达判断，但删除 action head 时必须同步改训练及在线代码，不能只改一个开关后沿用所有入口。

## 6. DFormer 替换方案与仓库组织建议

```text
RGB + Depth
    ↓
DFormer 多尺度特征图
    ↓
Token Adapter：选尺度 → 通道投影 → 二维位置编码 → 展平
    ↓
context C ← 目标条件
    ↓
现有 CoFL-S decoder ← 空间查询
    ↓
二维场
```

- 原版 DFormer `forward(rgb,depth)` 返回 `(outs,None)`；v2 返回特征 tuple，包装层统一接口。
- 可先测试原版 Tiny。四级通道 32/64/128/256；224×224 输入对应 56²/28²/14²/7² 特征图。
- 第一版可从 stride-16 的 14×14 特征映射到 768 维，形成 196 个 token。
- 显式加入二维位置编码：flatten 顺序本身不会告诉 Cross-Attention 图像空间位置。
- 旧 decoder 权重可以作为初始化候选，但新旧 context 分布不同，不保证冻结即可使用。
- 预处理必须匹配选定 DFormer 权重；米制深度与 encoder 输入规范分开保留，不能直接照搬 SigLIP/旧深度归一化。
- DFormer 性能、延迟、当前 PyTorch 兼容性尚未实测。

建议组织（尚未实施）：

```text
src/cofl/models/
  backbones/dformer.py
  adapters/spatial_tokens.py
  conditioning.py
  encoder.py
  policy.py
  cofl_s.py
  blocks.py
src/cofl/data/preprocessing.py
third_party/dformer/          # 选定源码、许可证、上游 commit 和修改说明
configs/rgbd_goal_field.yaml
```

不需要为未来合并原项目保留复杂兼容分支。初期保留包名，接通训练后再清理旧代码。

必须一起调整：

- `training/policy.py`：构建、预处理、加载。
- `training/data.py`：移除 SigLIP processor 重建。
- `training/module.py`：动作损失、视觉/文本优化器分组、checkpoint assets。
- `training/config.py`：改为新 encoder 与目标配置。
- `online/policy.py`：移除对 action head 的无条件调用，更新输入。
- checkpoint：保存新架构、预处理、完整 state_dict 和场几何配置。

### 目标坐标的区别

- 恒定机体相对目标，例如每帧前方 5m：可以将规则内置，但会持续向前，并不等于到达最初世界点。
- 固定世界目标：必须每帧转换为机体相对目标 `goal_ego`，需要位姿信息。
- 可将相对目标用 MLP 编成 goal token；确切条件注入方式待实现。

## 7. 原训练数据的含义与 Dijkstra 顺序

原数据是任务监督数据，不是单独 encoder 的预训练集。

输入：RGB-D＋指令；主要标签：流场；附加动作标签、mask、几何信息等。

当前仓库的流场标签生成顺序：

```text
仿真可通行几何 + 目标
    ↓
建立栅格图：距离 + 可选安全代价
    ↓
从目标运行 Dijkstra → 代价距离 + 前驱树
    ↓
势函数、负梯度、平滑与障碍处理
    ↓
监督流场
```

- Dijkstra 在**标签生成之前**运行，不是在预测流场后才找最短路。
- 图上的最小代价体现路径长度和可选安全偏好。
- 自由区方向来自势函数负梯度，幅值关联同一前驱树上的剩余几何路径长度 / v_norm。
- 后处理、模型拟合和轨迹积分没有严格全局最优保证。
- 当前部署推理以预测场积分轨迹，不在该控制链内重新运行 Dijkstra。
- 这是本地现有代码的结论；未逐项验证发布数据的历史生成配置完全一致。

## 8. 对 329 GiB 数据是否有用的最终结论

用户关心的是：能否直接训练自己的新约束任务，而不是“理论上能否用于预训练”。

**结论：不能把原数据中的地面语言导航流场直接当作变形无人机多点约束、新目标函数的正确监督。没有必要为了当前目标先下载全量数据。**

- 仅换 encoder，任务不变：原监督可以继续用。
- 换成用户的新优化问题：需要新规则下的监督标签。
- RGB-D 观测及原任务预训练仍有潜在用途，但这不是用户当前要的直接可用数据。
- 发布 RGB-D 不等于完整场景几何，不能假定仅靠这些观测就能重新做完整碰撞与最优规划。
- 删除语言改用坐标时，需要与标签匹配的真实目标元数据；不能把截断轨迹终点直接当成最终目标。

之前回答一度把“可用于预训练”泛称为“可以用”，已明确纠正。后续请直接围绕新任务监督回答，不再混淆。

## 9. 多点耦合约束：尚待确定的关键设计

原场为 `v(p)`，各空间查询独立；用户可能需要对多个机体点的联合状态 P 输出联合运动 U：

\[
U^*(P)=\arg\min_U J(P,U),\qquad c(P,U)=0,\quad h(P,U)\leq0.
\]

逐点查询再拼接不能自动保证机体几何、变形和碰撞约束。

待比较的路线：

1. 保留二维空间引导场，由后端约束优化器联合求解实际控制。
2. 输入机体位姿/形态，在构型空间预测联合运动；可能需要改 decoder 输出或增加联合交互/约束层。

**没有决定必须改成哪一种；不能把“直接保留 decoder 即可满足所有约束”当作已证实结论。**

## 10. 数据生成 pipeline 已在仓库中

无需从零编写整个 Habitat → 流场 → 原生训练数据流程。

- `configs/generation/sector_r2r.yaml`、`sector_rxr.yaml`：现有 recipe。
- `generation/sector/pipeline.py`：episode 生成适配。
- `generation/sector/worker.py`：独立运行环境 worker。
- `generation/sector/fields/sector_gt.py`、`global_geodesic.py`：标签核心。
- `generation/runner.py`：分片写入、校验、恢复和集合发布。

现有 R2R recipe 需要自行准备：兼容 Habitat 环境、场景/navmesh、episode、GT 回放动作、细粒度指令对齐及任务扩展。
`uv sync` 不自动安装完整 Habitat 或下载这些资源。文档记录过 Python 3.7 + Habitat-Lab/Sim 0.1.7 的单 episode 验证，不代表当前机器已配置，也不是全量复现保证。

资源就绪、路径配置正确后的小规模命令：

```bash
uv run --locked cofl data generate \
  --config configs/generation/sector_r2r.yaml \
  --output datasets/cofl-s-preview --limit-units 1
uv run --locked cofl data validate datasets/cofl-s-preview
```

当前是先生成并保存，再训练；不是已经实现训练时在线实时生成。
部分集合用于训练时还需显式允许 partial，并准备正确的 train/validation split。
新无人机任务可复用框架，但目标采样、约束规划、标签生成和必要的字段/坐标规范仍要适配。

## 11. 本机已经做过的工作与问题

- uv 0.12.10、Python 3.12.12、项目依赖已安装，`.venv` 已建立。
- `source "$HOME/.local/bin/env"` 因文件不存在报错；当前终端可用 `export PATH="$HOME/.local/bin:$PATH"`。
- 曾构建并成功启动 Studio；临时 Node.js 用于构建。当前是否仍运行未检查。
- 用户已登录 W&B；这不等于登录 Hugging Face。
- 运行正式训练时 SigLIP 预训练资产已下载，但因 `datasets/cofl-s` 不存在报错，没有开始实际训练。
- 没有进行 DFormer 集成、数据生成、无人机约束训练或 remote 迁移。

## 12. 下一步讨论应优先回答的问题

1. 无人机状态：平面位姿还是三维位姿？形态用哪些自由度描述？
2. 多个点的定义：机体固定点、连杆端点还是可独立控制部件？
3. 联合约束和优化目标的具体公式，以及动态可行性要求。
4. 希望网络输出引导场，还是直接输出联合可执行运动？
5. 目标是恒定机体系点还是固定世界点？
6. 训练场景/仿真器来源，是否继续使用 Habitat，如何表达无人机体积与变形。
7. 先做一个小规模自生成数据的闭环，再决定 DFormer 版本和大规模训练。

回答偏好：中文，直接、精确，围绕当前问题；区分已核实代码、设计建议和未验证假设。用户目前请求的是开发上下文整理，不是立即启动上述改造。
