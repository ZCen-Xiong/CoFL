# CoF-M 使用手册：配置、生成、训练与部署

更新日期：2026-11-18。本手册是用户操作入口；实现与硬件资料见
[仓库上下文](repository-context.md)。所有命令从仓库根目录执行，默认已激活
`cofl-navigation`，因此使用 `python`。Habitat 使用独立解释器。

## 1. 当前任务定义与可用阶段

当前先生成**质点二维向量场 `F(x,y)=(Fx,Fy)`**，复用仓库中 CoFL-S 的测地
距离、安全代价、势函数、梯度平滑及场幅值计算。差异是：无语言、手动起终点、
低相机高度，以及可配置的墙面净空（默认 0.02m）。

不在生成质点场时按整机十字膨胀障碍，不使用 q 切片或 Fq。之后从 COG 附近
质点取流线、形成端点规划和耦合控制，是后续阶段，当前尚未实现。

| 阶段 | 状态 |
| --- | --- |
| MP3D 楼层导入、低高度 RGB-D | 已实现 |
| 手动选点、CoFL-S 数值算法生成质点场 | 已实现 |
| 窄段筛选、RGB-D 与二维局部标签导出 | 已实现 |
| COG 附近多质点流线与端点耦合规划 | 尚未接通 |
| DFormer 训练、在线策略与真机部署 | 尚未接通；按要求暂停模型运行 |

## 2. 配置与重新导入场景

已准备的 MP3D 示例为 `17DRP5sb8fy`，源资产在
`artifacts/cof-m/mp3d-example-assets/`。完整 MP3D 场景需自行取得数据访问许可，
本命令不下载资产。Habitat 压缩场景包约 15GB，单场景示例目录约 158MB；
300GB 足够起步，长期容量主要取决于采集帧数。
[官方数据表](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)、
[MP3D 资产说明](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset)。

当前可用的**零半径质点 navmesh** 导入结果在
`artifacts/cof-m/matterport-point-020/`，相机相对楼面 0.2m。
旧 `matterport-preview/` 和 `matterport-preview-new/` 使用过非零半径侵蚀，
保留为历史产物，不能直接当成只有 2cm 净空的新场景使用。

重新导入示例（输出目录必须为空，每次使用新目录）：

```bash
python -m cofl.generation.matterport_scene \
  --scene artifacts/cof-m/mp3d-example-assets/17DRP5sb8fy/17DRP5sb8fy.glb \
  --scene-dataset artifacts/cof-m/mp3d-example-assets/mp3d.scene_dataset_config.json \
  --floor-y 0 --height 0.2 --resolution 0.01 \
  --habitat-python .habitat/cofm/bin/python \
  --output artifacts/cof-m/matterport-point-new
```

| 参数 | 含义 |
| --- | --- |
| `--floor-y` | Habitat Y-up 世界中的楼面高度 |
| `--height` | 相机光学中心相对楼面高度；当前还用于 navmesh 净高条件 |
| `--resolution` | 占据栅格间距；示例 1cm，CLI 默认仍为 2.5cm |
| `--floor-tolerance` | 地面扫描高度容差，默认 ±0.1m |
| `--crop-habitat XMIN XMAX ZMIN ZMAX` | 可选房间范围，Habitat 水平 X/Z 坐标 |

输出 `free.npy`、`raster.json`、`scene.json`、`studio.yaml` 和 `import.log`。
`scene.json` 是后续程序的场景清单，包括源资产位置、栅格坐标和相机配置。
**它是导入结果，不是导入命令的输入配置。** 只编辑 `flight_height_m` 不会改变
相机，网页读取 `config.camera_height`；手改后者也不会重建栅格和旧 RGB-D。
改变高度、分辨率或裁剪后应重新导入，并重新校验端点。

## 3. 设置质点净空和其他参数

在要使用的 `studio.yaml` 顶层配置，例如：

```yaml
device: cpu
cofm_point_clearance_m: 0.02
cofm_field_resolution_m: 0.025
cofm_safety_band_m: 0.02
cofm_safety_cost_weight: 5.0
scenes:
  mp3d-floor:
    path: scene.json
    label: Matterport point field / camera 0.2m / clearance 2cm
    domain_id: cofm
```

| 参数 | 控制对象 |
| --- | --- |
| `cofm_point_clearance_m` | 质点到障碍/规划边界的硬净空，默认 2cm |
| `cofm_field_resolution_m` | 质点规划和场网格间距，默认 2.5cm |
| `cofm_safety_band_m` | CoFL-S 安全代价带宽，默认 2cm |
| `cofm_safety_cost_weight` | 安全带内软代价权重，默认 5 |

硬净空与软代价分开。增大软代价带会偏好远离墙的路线，不会改变硬可行域。
navmesh 重建半径为 0，避免导入时先膨胀，再在 teacher 中重复加入净空。

**2cm 是配置的几何净空阈值，不是每排箭头距真实墙面的精确测量。**
占据单元有面积、采样有离散误差，当前用保守距离下界检查；示例 1cm 占据格
仍带来毫米至厘米级误差。场网格和箭头抽样也会让图上第一排箭头更远。
现在没有完整机体半径/臂长造成的数十厘米膨胀。需要更精细时减小占据与场
分辨率，不能把显示间距当作额外 clearance。

## 4. 打开网页与手动选点

已有正确质点场景可直接启动：

```bash
python -m cofl app \
  --config artifacts/cof-m/matterport-point-020/studio.yaml \
  --no-browser --port 8788
```

打开 <http://127.0.0.1:8788> → **cof-m Playground**。页面不会自动随机生成
起终点，先选择“地图选起点/终点”再点击地图，随后点击生成 teacher。
坐标按实际场分辨率对齐，只检查质点净空，不再要求全形态停泊。

主列表 `artifacts/cof-m/habitat-train-preview/studio.yaml` 也已将新质点场景
排在第一项，后面保留程序化测试场景。修改 catalog 后重启服务才能载入。
更换 `--output` 目录后，使用新目录的 `studio.yaml`，不要继续加载旧路径。

RGB/深度按钮切换观测；网页相机固定水平朝局部 +X，俯仰为零。路径滑条
浏览的是 **Dijkstra 质点参考路径**，并不是已经实现的 COG 质点流线积分。
点击“更新当前位置 RGB-D”采集该点图像。网页局部图是全局质点场的扇区展示；
批量标签会另外应用下节中的 RGB-D 可见性处理。

端口占用时，先在原终端 Ctrl+C 或点 **Shut down Studio**；关闭浏览器不会
停止服务。查看占用者可用 `ss -ltnp 'sport = :8788'`，确认进程后再
`kill -TERM <PID>`。前端源码变更后需 npm/Node 构建，已有资源时无需重复构建：

```bash
python -m cofl app \
  --config artifacts/cof-m/matterport-point-020/studio.yaml \
  --build --no-browser --port 8788
```

## 5. 手动任务与批量 RGB-D 采集

界面导出的 `artifacts/cof-m/studio-tasks/<id>/task.json` 可以直接作为采集任务。
也可编写 JSON 列表，坐标需在所选新场景中校验：

```json
[
  {"scene_id": "mp3d-floor", "start": [3.7, 5.6], "goal": [6.0, 0.1]}
]
```

新质点场景目录中已有这组 `tasks.json`。采集：

```bash
python -m cofl.experiments.matterport_routes \
  --config artifacts/cof-m/matterport-point-020/studio.yaml \
  --tasks artifacts/cof-m/matterport-point-020/tasks.json \
  --max-width 1.1 --min-length 0.3 --max-frames 200 \
  --camera-heading path \
  --output artifacts/cof-m/point-rgbd-new
```

批量采集默认筛选双侧宽度 ≤1.1m、连续长度 ≥0.3m 的窄段，只采窄段及前后
0.5m 路径窗口；不修改用户指定的端点。网页默认不强制窄段，可选勾选筛选。
只贴一面墙不算双侧窄通道。检查 `dataset.json` 的 accepted/rejected。
相机默认沿参考路径取景，`--camera-heading fixed` 可固定朝 +X。

输出格式为 `cofm_point_rgbd_routes_v1`：

```text
dataset.json
teachers/<id>/teacher.npz              # field[Ny,Nx,2]、价值、几何距离及掩码；无 q 轴
teachers/<id>/path.npy                 # [N,2] 参考路径
teachers/<id>/task.json
route-0000/poses.json                  # state_xy、goal_ego、heading_rad、Habitat 位姿
route-0000/000000-rgb.png               # uint8 [240,320,3]
route-0000/000000-depth.npy             # float32 米制光轴 Z 深度，无效为 0
route-0000/000000-valid.npy             # bool 有效深度
route-0000/000000-field.npz             # query[N,2]、field[N,2]、terminal[N]
```

RGB-D 对齐，水平 FOV 90°，裁剪范围 0.05–15m。局部 query 为
`[theta/(pi/4), range_m/5]`。局部标签复用 CoFL-S 的深度可见性、势函数、
平滑及场幅值函数，XY 向量和目标转到相机前/左坐标。
非终点幅值为同一目标根前驱树上的剩余几何距离 / 5，不是单位方向，也不是
飞控的物理速度。不可达与 terminal 分开。

## 6. 算法边界、训练和部署

复用的是仓库中 CoFL-S 的数值函数，不是重新命名旧 `(x,y,q)` teacher：

```text
零半径 navmesh / 地面占据 + 用户终点
 → 质点净空与 CoFL-S 安全代价
 → 目标根 Dijkstra（代价距离与同树几何距离）
 → 势函数、梯度与平滑
 → 二维向量场 F(x,y)
 → 之后再做 COG 附近质点流线与端点规划（待实现）
```

当前仍采用此前同意的 MP3D 地面占据平面近似；完整三维障碍、飞行器厚度、
动力学与端点耦合不属于该质点场的保证。全局 teacher 可以使用遮挡后的几何，
单帧 RGB-D 不一定能唯一推断全局最优分支。

`cofl train` 目前仍只支持原版 CoFL/CoFL-S，不能直接读取上述实验数据。
DFormer 原型仍有旧三分量/q 接口，**尚未按本次二维质点任务重新适配，也未运行**。
下一步先接通 COG 多质点流线方案，再明确训练数据、条件、二维输出、损失和
checkpoint；之后才是仿真闭环及 AMB 飞控/ROS 部署。没有可直接运行的 CoF-M
训练或真机部署命令，不能拿原版语言导航命令代替。

验证质点场与原 CoFL-S 数值一致、墙边质点可行、净空可配置及局部标签：

```bash
python -m pytest -q reproduce/test_point_field.py \
  reproduce/test_cofm_studio.py reproduce/test_matterport_routes.py
```

### 约 65cm 的真实 MP3D 窄口测试

`artifacts/cof-m/mp3d-gap-065-test/` 来自 `17DRP5sb8fy`，未修改家具网格。
裁剪地图约 2.19×2.59m；相机高 0.2m，质点净空 0.02m。
在该高度用相反方向的 Habitat 深度射线测得横断面约 0.6503m，
相邻断面约 0.659m；这是家具边缘形成的短窄口，不是恒宽走廊，
也不代表其他高度的净宽。详见目录内 `gap-measurement.json`。

```bash
python -m cofl app --config artifacts/cof-m/mp3d-gap-065-test/studio.yaml --no-browser --port 8788
```

先停止占用 8788 的旧 Studio，再启动并刷新网页。场景也已加入原
`habitat-train-preview/studio.yaml` 的列表，名称含 `approx 65cm gap`。
在地图选择起点 `(0.900, 0.075)m`、终点 `(1.500, 0.075)m`，
选择质点 teacher 生成操作。已验证路径可达，且直线通过测量横断面。
`test-task.json` 保存这组点；`teacher-preview.json` 和 `tasks/` 保存预生成结果，
`map-preview.png` 是地图预览，`rgbd/` 有起点、窄口及终点三个 RGB-D 观测。
此次未运行 DFormer。
