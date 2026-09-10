# 感知模块（YOLO）

主文档：`使用说明.md`。

```
yolo/
  使用说明.md
  数据集展示.md
  仿真/     # 界面调用；124 类中文分割 + 可选 tiny19 小件补检
  实机/     # RealSense；113 类英文分割 + 全量 dataset/
```

仿真推理：

```bash
python tiaozhanbei/yolo/仿真/infer_seg_annotations.py --sample 0001 --save-masks
```
