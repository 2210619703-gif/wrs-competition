# 决策模块（XH-202607）

**主入口（连接智能体）：** `python agent_demo_ui.py`  
**使用说明：** `使用说明.md`

```
decision/
  agent_demo_ui.py        # 主入口，连接 Dify 智能体
  启动Dify环境.bat        # 拉起本机 Dify（软件见使用说明第 1 节）
  agent/                  # 连接智能体的内部调用
    demo_voice_agent.py
    run_batch.py
  workflows/              # 导入 Dify 的 yml
    workflow.yml
    build_workflow.py     # 改 workflow_code 后重新生成 yml
  workflow_code/          # 决策核心（智能体与本地共用）
  offline/                # 未连接智能体时的本地规划
  simple_dataset.json
  使用说明.md
```
