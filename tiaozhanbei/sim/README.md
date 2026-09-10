# 执行模块（当前为仿真）

读决策任务 JSON，在 Panda3D / WRS 里做抓取与放置。真机代码以后再放进本目录。

**主入口（标准抓放）：**

```bash
python tiaozhanbei/sim/run_agent_pick_place_side_sim.py --task tiaozhanbei/tasks/delivery_agent_auto.json
```

**螺丝入格：**

```bash
python tiaozhanbei/sim/ec/run_ec_bin_sim.py --task tiaozhanbei/sim/ec/tasks/test03.json
```

展示时仍从决策界面拉起：`python tiaozhanbei/decision/agent_demo_ui.py`

详见 `使用说明.md`。
