# CoFL-M Habitat 训练场景预览

2026-11-15 已生成 6 个场景、96 帧纯深度观测。train/val 按场景划分，种子不重叠。

同日已新增独立的 **CoFL-M → CoFL-M Playground**，显示第一人称深度、随机/手选停泊点、teacher 场和路径。当前使用方法见 [新入口与原版 CoFL-S 教程](cofl-m-studio.md)。新任务请使用 CoFL-M 入口。

| Split | 类型 | Seed | 深度帧数 | 最小表面净距 |
| --- | --- | --- | --- | --- |
| train | 45 柱 | 100 | 16 | 0.650456m |
| train | 25 柱＋8 墙 | 101 | 16 | 0.717306m |
| train | 20 墙 | 102 | 16 | 0.670178m |
| train | 45 柱 | 103 | 16 | 0.655385m |
| val | 45 柱 | 200 | 16 | 0.655864m |
| val | 25 柱＋8 墙 | 201 | 16 | 0.652067m |

全部为 16×10m 场地、3m 全高垂直障碍、无圆环，固定规划/相机高度 z=1.2m。深度为 320×240 float32 米制光轴深度，附 bool 有效性掩码、相机位姿和几何审计。每场景沿现有中心采样线采集 16 帧；尚未扩展到任意可行构型采样。

这是训练使用的场景与观测准备集，**没有正式流场监督标签，尚不是可直接用于现有 CoFL-S 训练入口的 CoFLDataset**。manifest 显式保存 `labels_included=false` 和 `supervised_training_ready=false`。

## 生成与产物

```bash
.venv/bin/python -m cofl.experiments.habitat_scene_set \
  --output artifacts/cofl-m/habitat-train-preview \
  --habitat-python .habitat/coflm/bin/python
```

输出目录非空时拒绝覆盖；复现需使用新目录。产物根目录：`artifacts/cofl-m/habitat-train-preview/`。

```text
manifest.json                   # split、场景、种子、观测数量与审计
studio.yaml                     # 复用已有 Studio 的六场景目录
train/train-poles-100/           # 其他场景目录同理
  obstacles.glb
  obstacles.stage_config.json
  scene.json
  depth/000000.npy
  valid/000000.npy
  preview/000000.png
  habitat.log
studio-poles.png                # 本次已有 Studio 的浏览器截图
studio-mixed.png
studio-verification.json        # 本次资源和显示检查
```

## 使用已有可视化

仓库已有 CoFL Studio，现在有独立 CoFL-M 入口，可直接使用这些场景采集第一人称深度、验证停泊点并生成和显示 teacher；不依赖原 CoFL-S 的语言、episode 或 checkpoint。

```bash
.venv/bin/python -m cofl app \
  --config artifacts/cofl-m/habitat-train-preview/studio.yaml \
  --no-browser --port 8788
```

打开 <http://127.0.0.1:8788>，选择 **CoFL-M → CoFL-M Playground**，然后在左侧场景菜单选择 `CoFL-M train / ...` 或 `CoFL-M val / ...`。页面自动生成有效停泊起终点、采集真实深度并展示 teacher。可手动选点、查看形态切片和下载标签。具体操作见新教程；旧版本缓存需 Ctrl+Shift+R。

此前展示仅借用了 BEV 的 GLB 浏览器。新入口已经取消语言/旧模型控件，使用独立任务 API；teacher 保存到 `artifacts/cofl-m/studio-tasks/`，尚不是网络预测。

当前 Studio 使用 CPU 配置，浏览场景不会加载训练权重；Habitat 深度已离线渲染，无需持续运行模拟器。关闭时可使用页面左下角 **Shut down Studio**。本次后台服务的日志和 PID 分别在产物根目录的 `studio.log`、`studio.pid`。
