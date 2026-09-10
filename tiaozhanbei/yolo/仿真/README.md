# 仿真感知

124 类中文分割。权重：`best.pt`。**不要使用 `../实机/best.pt`。**

场景图来自 `tiaozhanbei/dataset_learn/`，本目录不再存放采集脚本。

```bash
python infer_seg_annotations.py --sample 0001 --save-masks --pretty
python annotations_to_vision.py --ann outputs/0001.json
```

训练与状态头见上一级 `使用说明.md`。
