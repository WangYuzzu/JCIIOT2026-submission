# JCIIOT 2026 Industrial Embodied AI Submission

本仓库是 JCIIOT 2026 工业具身智能挑战赛的完整可复现提交。方案将
**SOP 知识生成、受约束 LLM 规划、A\* 安全导航、统一多任务 Transformer
行为克隆（BC）、物理接触/抬升校验和轨迹审计**组合为一套可解释的移动操作系统。

## 最终结果

| 关卡 | 得分 | 轨迹帧 | 碰撞帧 | 抓取 |
|---|---:|---:|---:|---|
| L1 | 10/10 | 304 | 0 | BC 接触并抬升 |
| L2 | 15/15 | 344 | 0 | BC 接触并抬升 |
| L3 | 20/20 | 360 | 0 | BC 接触并抬升 |
| L4 | 25/25 | 367 | 0 | BC 接触并抬升 |
| L5 | 30/30 | 974 | 0 | 三次 BC 接触并抬升 |
| **合计** | **100/100** | **2,349** | **0** | **7/7 分支通过** |

每一帧均记录机器人底盘位姿、27 个关节角和可移动物体的 7 维位姿。最终
checkpoint 是一个 12.9 MB 的普通 Git 文件，不依赖 Git LFS。

## 三步复现

```bash
git clone https://github.com/WangYuzzu/JCIIOT2026-submission.git
cd JCIIOT2026-submission/JCIIOT

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ./robomimic -e ./robosuite -e .

python team_submission/verify_submission.py
```

期望最后一行为：

```text
TOTAL: 100/100 PASS
```

离线验证不启动 MuJoCo，也不需要 API。正式仿真和 Streamlit 复现步骤见
[提交说明](JCIIOT/team_submission/README.md)。API key 只通过环境变量提供，
仓库不包含任何密钥。

## 模型与 API

| 角色 | 模型 / 资产 | 用途 |
|---|---|---|
| 文本规划 LLM | 智谱 `glm-5.2` | 将任务和 SOP 上下文转换为受限 JSON 动作计划 |
| 视觉模型 VLM | 智谱 `glm-5v-turbo` | 仅用于离线解析 SOP 文档中的图片 |
| 连续控制策略 | `jciiot_unified_task_heads_v16_deploy.pth` | 执行七种双臂近场抓取条件 |

LLM 和 VLM 均通过智谱开放平台的 OpenAI-compatible endpoint
`https://open.bigmodel.cn/api/paas/v4` 调用。界面中的 **OpenAI API** 表示接口
兼容模式，并不表示使用了 OpenAI 模型。正式报告轨迹使用 `glm-5.2`，规划
temperature 为 `0.1`；VLM 不参与最终轨迹验证和离线回放。完整变量示例见
[`team_submission/.env.example`](JCIIOT/team_submission/.env.example)。

## 方法概览

```text
DOCX SOP + task config + semantic map
                  │
                  ▼
       可追溯知识 Markdown + 事实校验
                  │
                  ▼
        LLM 结构化计划（受限 JSON）
             ┌────┴────┐
             ▼         ▼
       A* 底盘导航   统一 Transformer BC
             └────┬────┘
                  ▼
       接触 / 抬升 / 碰撞 / 目标审计
                  ▼
       result + trajectory + GIF + score
```

主要创新点：

1. 将 SOP 文本、官方任务配置和实时场景语义图组成带 provenance 的知识闭环；
2. 使用一个共享 Transformer 主干和七个轻量任务动作头，在单 checkpoint 中覆盖五关；
3. BC 抓取必须通过双侧指垫接触和真实抬升，之后才允许使用官方运输 attachment；
4. 将“训练成功、执行成功、评分成功”分别审计，避免仅凭 `success=true` 误判。

## 提交材料入口

- [完整技术报告（Markdown）](JCIIOT/team_submission/TECHNICAL_REPORT.md)
- [完整技术报告（PDF）](JCIIOT/team_submission/TECHNICAL_REPORT.pdf)
- [英文论文版技术报告（PDF）](JCIIOT/team_submission/TECHNICAL_REPORT_EN.pdf)
- [英文论文 LaTeX / Overleaf 源码](JCIIOT/team_submission/paper/)
- [LLM/VLM Prompt 设计与输入构成](JCIIOT/team_submission/PROMPT_DESIGN.md)
- [复现与验证记录](JCIIOT/team_submission/VALIDATION.md)
- [训练与数据再生成说明](JCIIOT/team_submission/TRAINING.md)
- [五关轨迹和机器可读评分](JCIIOT/team_submission/evidence/)
- [五关 GIF 演示](JCIIOT/team_submission/demos/)
- [统一 BC checkpoint](JCIIOT/team_submission/models/jciiot_unified_task_heads_v16_deploy.pth)
- [外部参考资产说明](JCIIOT/team_submission/ASSETS.md)

## 合规说明

方案基于官方功能提交 `fa0eaef`。官方后续 `01032e8` 仅更新排行榜 README，
没有改变评分、场景或接口。赛事禁止修改的 `app.py`、`src/robot_agent/core/`、
`src/robot_agent/environments/` 和 `knowledge/task_config.json` 均与功能基线一致。
参赛者改动范围和第三方项目引用详见技术报告。

许可证继承自官方项目，见 [LICENSE](JCIIOT/LICENSE)。
