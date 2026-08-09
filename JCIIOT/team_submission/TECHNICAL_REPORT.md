# JCIIOT 2026 移动操作机器人方案技术报告

**提交版本：** 2026-08-09
**基线：** 官方仓库提交 `fa0eaef`
**范围：** L1–L5 轨迹生成、规划、双臂抓取、放置与可复现实验

## 摘要

本方案将自然语言任务拆成“知识生成—符号规划—安全导航—行为克隆抓取—物理校验—放置—轨迹审计”七个环节。语言模型不直接输出关节控制量，而是从 DOCX SOP、赛题配置和场景语义映射中生成受约束的 `move / pick_up / place_down` JSON 计划；A* 在二维占据栅格上生成移动底盘路径；一个带 7 维任务条件的低维 Transformer BC 统一承担 L1–L5 的七个训练分支；执行器在每次抓取后检查双侧指垫接触和物体抬升，再允许搬运。

最终提交的唯一 BC checkpoint 为 12,928,025 字节，共 3,210,032 个参数：一个共享 Transformer 主干及七个轻量任务动作头。共享训练使用 336 条完整成功示范和 144 个真实专家轨迹中的接触闭合窗口；随后仅解冻特定任务头，分别校准 L5-center、L2 和勘误后的 L3。对 2026 年 8 月 9 日官方勘误后的五关重新执行和复算，提交证据达到 **100/100**，共记录 **2,349 帧**，碰撞标志为 **0 帧**。这一结果是固定官方场景上的确定性验证，不等同于未公开扰动下的统计成功率。

L3 的官方勘误已完整纳入最终训练：重新采集 48/48 条成功专家轨迹，目标为 `aux_input_1` 的 `blue_tote_b01_near_right`，停车姿态与正式无碰撞导航一致。最终 L3 使用 BC 形成双臂四指接触并实际抬升 0.123 m；运行时不存在“抓取失败后 attachment 恢复”的路径。全部七个分支都必须通过相同的真实接触与抬升门槛。

## 1. 合规边界

本提交以官方最新代码为基线，不修改赛事禁止修改的文件：

| 官方边界 | 状态 |
|---|---|
| `app.py` | 与官方 `fa0eaef` 一致 |
| `src/robot_agent/core/` | 与官方一致 |
| `src/robot_agent/environments/` | 与官方一致 |
| `knowledge/task_config.json` | 与官方一致 |

参赛者改动集中在 `src/robot_agent/skills/`、独立任务子进程、`knowledge/robot_params.json`、可生成的知识 Markdown、训练/审计工作流和 `team_submission/`。轨迹保留机器人世界坐标、27 个关节角和全部可移动物体的 7 维位姿，满足官网新增提交要求。

## 2. 总体架构

```text
DOCX SOP + task prompt + task_config + semantic map
                         │
                         ▼
              生成并校验知识 Markdown
                         │
                         ▼
              LLM 结构化任务规划（JSON）
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       A* 底盘导航             统一 Transformer BC
       障碍膨胀/近场姿态        双臂连续动作
             │                       │
             └───────────┬───────────┘
                         ▼
             接触、抬升、碰撞与目标校验
                         │
                         ▼
            result / trajectory / score / GIF
```

关键设计是职责解耦：LLM 负责语义理解和动作序列，不碰连续控制；BC 只在经验证的近场姿态执行抓取；确定性代码负责地图、对象身份、物理门槛和评分字段。这样既保留大模型对 SOP 的理解能力，又避免把数值运动控制交给不稳定的文本生成。

## 3. LLM 与 SOP 知识生成

### 3.1 给 LLM 的上下文如何构成

规划上下文由四类信息合并：任务原始文本、当前关卡配置、场景动态物体映射、知识库 Markdown。L1 的基础知识还包含 `command_examples.md`、`pick_operation.md`、`place_operation.md` 和 `constraints.md`；它们由 `read_document.py` 读取并注入规划提示。执行子进程初始化仿真后读取当前场景的真实物体名/语义端口，再由官方 `task_config.json` 限定允许对象、数量和源/目标，避免仅凭 DOCX 可见标签猜内部端口。

### 3.2 DOCX → Markdown

`generate_sop_knowledge.py` 读取五份 DOCX 的段落、表格和内嵌图片；图片由 VLM 描述，文本模型将证据整理成可执行 SOP。随后程序把官方任务配置与语义地图解析出的事实覆盖到单独的“Planner-Critical Facts”区，并执行以下一致性检查：

1. 源端口、目标端口必须在当前语义图存在；
2. 物体必须属于官方允许列表；
3. 数量大于 1 时，每个物体都必须有完整的四步循环；
4. `pick_up` 与 `place_down` 必须沿用同一精确物体名；
5. 旧版别名不得进入最终计划。

生成物、DOCX SHA-256、模型名称和执行事实记录在 `team_submission/knowledge/generated_sop_manifest.json`。8 月勘误后，L3 明确改为 `aux_input_1 → output_5` 的蓝色箱，L5 改为 `input_1 → aux_output_1` 的三个白边箱。

### 3.3 规划约束与容错

LLM 输出限定为 JSON schema，动作集合只有 `move`、`pick_up`、`place_down`。运行时会规范化模型偶尔产生的可读工位名，并用官方配置做最终守卫；多物体任务保留对象列表，不再在子进程边界意外压缩成一个标量。所用文本模型是 OpenAI-compatible 接口上的 `glm-5.2`；API key 仅通过环境变量传入，从未写入仓库。视觉模型只用于重新生成 SOP 图片描述，不参与最终轨迹回放或离线评分。

## 4. 导航、抓取与放置

### 4.1 安全导航

移动模块从语义地图和占据栅格解析工位中心，使用 A* 生成底盘路径，并在搜索前按机器人足迹膨胀障碍。到抓取工位前，先到无碰撞预接近点，再旋转到对应 BC 训练朝向；离开工位保留安全后退段。目标匹配先做精确匹配再做兼容匹配，避免 `aux_output_1` 被错误解析成 `output_1`。

### 4.2 一个统一的任务条件 BC

最终策略基于 robomimic/PyTorch 的低维 Transformer BC。输入包括双臂末端位置/四元数、两侧夹爪位置、时间步和 7 维 one-hot `bc_task_id`；输出为 20 维双臂连续控制动作。任务条件显式区分：

- L1 蓝色空心箱；
- L2 绿边箱；
- 勘误后的 L3 蓝色周转箱；
- L4 蓝色空心箱；
- L5 back / center / front 三个抓取分支。

模型不是七个 checkpoint 的打包：七个任务共享同一个 3,174,052 参数 Transformer/编码器，只增加七个线性动作头，共 35,980 个参数；checkpoint 总参数为 3,210,032。12.9 MB 小于官方约 133 MB checkpoint 的主要原因是本方案不含 RGB 视觉编码器，只使用状态向量；它不是被截断的模型。所有关卡加载同一个文件，场景参数只选择条件向量、对应动作头、近场姿态和 rollout horizon。

### 4.3 数据构成与专家轨迹

专家轨迹不是赛事样本 HDF5 的简单复制。数据采集器在 MuJoCo 中以解析式分阶段双臂控制器生成“接近—对齐—闭合—保持/抬升”动作，并对底盘 x/y/yaw、阶段时长和动作施加小扰动；只有通过双侧指垫接触、抬升高度和无碰撞门槛的 episode 才进入训练集。

| 分支 | episode | time step | train/valid |
|---|---:|---:|---:|
| L1 | 48 | 15,015 | 40/8 |
| L2 | 48 | 22,789 | 40/8 |
| L3（勘误后蓝箱） | 48 | 23,819 | 43/5 |
| L4 | 48 | 24,277 | 39/9 |
| L5 back | 48 | 24,705 | 39/9 |
| L5 center | 48 | 32,596 | 39/9 |
| L5 front | 48 | 31,659 | 39/9 |
| **完整轨迹合计** | **336** | **174,860** | **301/35** |

另从真实专家序列中裁切 144 个接触闭合窗口、16,752 个时刻，强化夹爪闭合这一在完整轨迹中占比较低的关键阶段。采用监督式行为克隆，不使用中间 reward，动作损失由 L2 主损失与较小权重 L1 项构成。早期单共享输出头在多任务训练中出现动作相互干扰；最终结构改为共享 Transformer 加七个任务动作头。共享模型从随机初始化训练后，仅校准失败任务对应的单一动作头，冻结共享主干和另外六头。checkpoint 选择标准不是最低 validation loss，而是七个分支严格“接触并抬升”测试的交集。

### 4.4 物理门槛与搬运 attachment

BC rollout 后，系统同时检查左右夹爪的两侧指垫是否接触目标碰撞几何，并要求物体实际抬升约 0.12 m。只有门槛通过后，官方已有的 transport attachment 才在长距离导航中保持刚性连接；到放置点后恢复物理、设定互不重叠的槽位和世界朝向，再释放。该 attachment 是“物理抓取已成功后的运输保持”，不是抓取替代方案。任一 BC 分支未通过接触/抬升门槛都会直接失败，最终代码不存在 attachment recovery。

## 5. 调优过程与形成的经验

1. **先区分执行成功与评分成功。** 早期独立 runner 显示 4/4 或 12/12，但 L2–L4 的事件 `source` 写成内部 BC alias，且部分轨迹有碰撞，按正式函数只有约 35/100。修复方式是内部仍用 alias 选策略，落盘事件恢复官方语义端口，并直接对 JSON 调用同构评分审计。
2. **一个模型需要显式任务身份和受控输出隔离。** 直接混合七组数据时，网络会把相似状态对应到不同动作；只加入 one-hot 条件仍会在共享输出层产生干扰。最终用共享 Transformer 加七个小动作头，把感知/时序表示共享与动作映射冲突分离。
3. **示范质量比盲目增加数量重要。** 抓取接近却未同时形成双侧接触的轨迹会让 validation loss 看似下降但物理失败。只保留通过接触、抬升、碰撞筛选的数据更有效。
4. **多物体放置必须考虑后续交互。** L5 首轮三个目标中心相同，后放物体会推走先放物体；仅做 x 方向偏移仍会因箱体长边重叠。最终将箱体旋转 90°并使用 `[-0.55, 0, +0.55] m` 三个槽位，三者均稳定落在 0.8 m 评分半径内。
5. **语义真值必须有单一优先级。** DOCX 是给人看的操作说明，内部端口来自场景映射，允许对象/数量来自官方 task config。将三者混成无优先级文本会导致旧勘误污染；当前以 task config 与实时场景覆盖生成文本，并保留 provenance。
6. **训练 loss 不能代替物理回归。** L3 头 epoch 3、5、7、9 通过，而 epoch 12 反而失去接触；最终选择最早稳定通过的 epoch 3。每次校准还逐参数审计，确认只有目标任务头发生变化。
7. **失败要可观察。** 每个技能输出输入参数、前置条件、预期输出、尝试次数；轨迹同时记录 BC 事件、碰撞和最终位姿，避免用一个 `success=true` 掩盖过程。

## 6. 最终结果与分析

下表由 `team_submission/verify_submission.py` 对 canonical JSON 复算。该脚本同时检查每帧的机器人位置、27 个关节角、可移动物体位姿、抓取事件源字段、离开源点、目标 XY 距离与碰撞标志。

| 关卡 | 步骤 | 轨迹帧 | 碰撞帧 | 抓取方式 | 得分 |
|---|---:|---:|---:|---|---:|
| L1 | 4/4 | 304 | 0 | BC 接触+抬升 | 10/10 |
| L2 | 4/4 | 344 | 0 | BC 接触+抬升 | 15/15 |
| L3 | 4/4 | 360 | 0 | BC 接触+抬升 | 20/20 |
| L4 | 4/4 | 367 | 0 | BC 接触+抬升 | 25/25 |
| L5 | 12/12 | 974 | 0 | 三次 BC 接触+抬升 | 30/30 |
| **合计** | **28/28** | **2,349** | **0** |  | **100/100** |

L5 三个最终目标距离分别为 0.583 m、0.494 m、0.521 m；L3 最终目标距离为 0.075 m，抓取事件明确记录真实成功且没有恢复事件。所有原始结果、轨迹、评分复算结果和 GIF 位于 `team_submission/evidence/` 与 `team_submission/demos/`。

### 6.1 优势

- 语言语义与连续控制分层，错误边界清晰；
- 单 checkpoint 覆盖七种抓取条件，部署和评审更简单；
- 抓取以后验物理量而非网络输出值作为成功依据；
- 评分字段、SOP 事实、场景端口和对象身份有自动一致性审计；
- 完整轨迹可在重启后离线回放和评分。

### 6.2 局限性

- 最终 100/100 是每关一次固定场景运行，尚未给出多随机种子置信区间；
- 当前按任务头选择动作，仍依赖正确的对象身份/任务条件；
- 低维策略依赖仿真状态真值，不能直接迁移到仅 RGB 的真实机器人；
- L5 CPU 回放较慢、峰值内存较高；
- LLM API 的规划延迟受外部服务影响，但轨迹验证与回放不依赖 API。

## 7. 新颖性声明

本方案并不声称发明 Transformer、BC 或 A*。创新点在于针对该赛题构建了一个可审计的组合系统：

1. **SOP provenance + 场景真值闭环。** LLM 生成的知识不是直接信任，而是由 task config 和实时场景事实覆盖、哈希留痕并在运行时再次守卫；
2. **共享主干、任务动作头的单 checkpoint 策略。** 七个几何/阶段不同的双臂抓取共享低维 Transformer 表示，用 one-hot 条件选择轻量动作头，兼顾参数共享与控制冲突隔离；
3. **以物理门槛驱动的混合执行。** BC、接触/抬升验证、物理成功后的长距离刚性保持和放置槽位共同组成状态机，每个跨层决策均进入轨迹事件；
4. **评分兼容性作为一等测试。** 训练成功、执行成功和得分成功分别审计，提交附带独立复算器而非只展示 UI 截图。

与仅靠 LLM 端到端控制相比，该架构把不可验证的连续决策压缩到离线训练策略；与每任务独立 checkpoint 相比，统一策略减少部署资产并展示了受控的多任务共享；与仅报告任务成功相比，完整事件/物理/评分证据使限制和恢复均可追踪。

## 8. 复现说明

### 8.1 安装与配置

```bash
git clone https://github.com/WangYuzzu/JCIIOT2026-submission.git
cd JCIIOT2026-submission/JCIIOT
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ./robomimic
python -m pip install -e ./robosuite
python -m pip install -e .
```

复制 `team_submission/.env.example`，只在本地填写 API key。正式 UI：

```bash
streamlit run app.py
```

命令行依次运行五关：

```bash
python team_submission/run_all_levels.py
```

无 MuJoCo、无 API 的离线完整性与评分检查：

```bash
python team_submission/verify_submission.py
```

生成回放：

```bash
PYTHONPATH=src:. python team_submission/generate_demos.py
```

checkpoint SHA-256：

```text
f8c7feb8047ad62f4e1e01f0e67886a0aa41f87781d486ae90e23164c37a7a5d
```

## 9. 第三方库与先前工作

- **robomimic**：示范数据格式、Transformer BC 训练和 checkpoint 运行；参见 Mandlekar 等人的[机器人离线示范学习研究](https://arxiv.org/abs/2108.03298)及[项目主页](https://robomimic.github.io/)。
- **PyTorch**：自动微分与神经网络训练。
- **robosuite / MuJoCo**：机器人环境、接触动力学和离屏渲染；MuJoCo 的设计见 Todorov 等人的[论文](https://doi.org/10.1109/IROS.2012.6386109)。
- **Transformer**：序列策略骨干源于 Vaswani 等人的[Attention Is All You Need](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)。
- **A\***：导航搜索基于 Hart、Nilsson、Raphael 的[最小代价路径启发式方法](https://doi.org/10.1109/TSSC.1968.300136)。
- **智谱开放平台**：规划采用官方文档列出的 [`glm-5.2`](https://docs.bigmodel.cn/cn/guide/models/text/glm-5.2)，SOP 图片理解采用 [`glm-5v-turbo`](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5v-turbo)。
- **Streamlit / OpenAI-compatible API / python-docx / h5py / NumPy / Pillow**：分别用于 UI、规划接口、SOP 解析、数据集处理、数值计算与 GIF 输出。

## 10. 提交清单

- 最终路径生成与技能代码；
- 唯一统一 BC checkpoint 及 SHA-256；
- 五关 canonical result/trajectory/verification JSON；
- SOP 生成物和 provenance manifest；
- 训练、转换、评估和审计脚本；
- 本技术报告 Markdown/PDF；
- 五关 GIF 演示；
- 环境变量示例、资产说明和一键复现命令。
