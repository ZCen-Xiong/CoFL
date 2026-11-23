# CoF-M 使用手册：配置、生成、训练与部署

更新日期：2026-11-22。本手册是用户操作入口；实现与硬件资料见
[仓库上下文](repository-context.md)。所有命令从仓库根目录执行，默认已激活
`cofl-navigation`，因此使用 `python`。Habitat 使用独立解释器。

## 1. 当前任务定义与可用阶段

当前先生成**质点二维向量场 `F(x,y)=(Fx,Fy)`**，复用仓库中 CoFL-S 的测地
距离、安全代价、势函数、梯度平滑及场幅值计算。差异是：无语言、手动起终点、
低相机高度，以及可配置的墙面净空（默认 0.02m）。

不在生成质点场时按整机十字膨胀障碍，不使用 q 切片或 Fq。之后从 COG 附近
质点取流线、形成端点参考已有 A1 离线原型；真实机构耦合控制和飞行闭环尚未实现。

| 阶段 | 状态 |
| --- | --- |
| MP3D 楼层导入、低高度 RGB-D | 已实现 |
| 手动选点、CoFL-S 数值算法生成质点场 | 已实现 |
| 窄段筛选、RGB-D 与二维局部标签导出 | 已实现 |
| 扇面双点选择、积分和 A1 参考接续 | 离线原型已实现；严格条件下未全程通过 |
| DFormer 二维场训练与跨帧评估 | 已完成真实 RGB-D 小样本实验；非部署模型 |
| AMB 控制器、ROS 与真实飞行 | 未接通 |

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

输出格式为 `cofm_point_rgbd_routes_v2`；旧 v1 标签需要重新采集：

```text
dataset.json
teachers/<id>/teacher.npz              # field[Ny,Nx,2]、价值、几何距离及掩码；无 q 轴
teachers/<id>/path.npy                 # [N,2] 参考路径
teachers/<id>/task.json
route-0000/poses.json                  # state_xy、goal_ego、heading_rad、Habitat 位姿
route-0000/000000-rgb.png               # uint8 [240,320,3]
route-0000/000000-depth.npy             # float32 米制光轴 Z 深度，无效为 0
route-0000/000000-valid.npy             # bool 有效深度
route-0000/000000-field.npz             # query[N,2]、field[N,2]、terminal[N]、world_xy、world_index
```

RGB-D 对齐，水平 FOV 90°，裁剪范围 0.05–15m。局部 query 为
`[theta/(pi/4), range_m/5]`。每个任务只生成一次世界流场；各帧读取同一
field，XY 向量和目标转到相机前/左坐标。深度可见性仅控制监督掩码，不再把
不可见区域当障碍重建势场；否则同一世界位置会随视角改变标签。
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
 → 当前扇面选双点、积分、参考接续（A1 离线原型）
```

当前仍采用此前同意的 MP3D 地面占据平面近似；完整三维障碍、飞行器厚度、
动力学与端点耦合不属于该质点场的保证。全局 teacher 可以使用遮挡后的几何，
单帧 RGB-D 不一定能唯一推断全局最优分支。

`cofl train` 目前仍只支持原版 CoFL/CoFL-S，不能直接读取上述实验数据。
CoF-M 已有独立实验入口，见第 7 节；没有接入原版语言训练 CLI。DFormer
输入为 RGB-D、目标 XY，输出二维质点场；q 输入与第三输出已移除。checkpoint
格式为 `cofm_point_rgbd_v2`，旧 `cofm_rgbd_v1` 构型权重会明确拒绝加载，需重新训练。
真实 AMB 部署尚未接通，不能把本次离线重放当成飞行闭环。

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
该窄口导入步骤未运行 DFormer；2026-11-22 的独立训练实验见第 7 节。


## 7. A1：连续帧采集、teacher、训练与双端点重放

本次用户已明确授权恢复 DFormer 小样本训练。完整产物在
`artifacts/cof-m/temporal-a-20261122/`。未下载或使用预训练权重。
这是同一个 MP3D 场景内的可行性实验，不是已验证的穿缝模型。

### 数据与帧关系

任务由 `configs/experiments/cofm_temporal_tasks.json` 手工指定：一条经过此前
约 65cm 家具短窄口的点路径，一条带约 90cm 栅格窄段的曲线路径。
从用户起终点生成 Dijkstra 点路径，以 0.25m/s、每 0.2s 重采样；每条最多
连续 24 帧，取窄段附近的连续窗口，不通过跳帧伪造 0.2s。相机沿平滑路径
朝向，并加入连续的小幅 yaw 扰动。真实 Habitat RGB-D 高 0.2m、HFOV 90°。
共 48 帧。相邻帧是已知 SE(2) 相对位姿的同一任务，目标世界坐标固定，
`goal_ego` 随相机位姿变化。点路径不是整机可行轨迹。

每任务只求一次固定世界 teacher。各帧标签中的 `world_index` 与 `world_xy`
标识相同的世界点，`query` 和 `field` 转到该帧相机坐标。可见性只裁剪监督，
不修改世界几何。深度是光轴 Z，所有二维规划仍采用当前地面占据近似。

### 网络与训练约束

DFormer-Tiny 编码 RGB-D，CoFL-S 查询解码器输出 `[B,N,2]`。网络没有历史
输入、语言、机身 q 或构型头；历史保存在部署参考和短时场缓存中。
输入缩放到 224×224，实验解码维度 64，全 encoder 与 decoder 都训练。
每五帧中索引余 3 的帧留作验证：38 训练帧、10 验证帧、28 个相邻训练对。
没有让包含验证帧的 pair 参与反向传播；验证帧用于选 checkpoint，因此
这里的数字是验证集指标，不是独立测试集或新场景泛化。

每次更新取一对相邻训练帧，每帧 512 个查询；256 个为两帧共同可见的相同
世界点，另 256 个为本帧自由采样。令 `R` 为机体到世界的 yaw 旋转，损失为：

```text
L = 10*MSE(F_pred, F_teacher) + 0.1*方向余弦误差
  + λ * [10*MSE(Rt*Ft, Rt1*Ft1) + 0.25*跨帧方向余弦误差]
```

跨帧项只比较相同世界点，方向项跳过 teacher 零幅值附近；平移用于查询点
对应，旋转用于向量对齐。不能直接比较两个扇面的同索引箭头。λ=0 为基线，
λ=2 为 A1 训练。AdamW lr=3e-4、weight_decay=1e-4、梯度范数裁剪 1，固定
种子 17；两组均训练 1200 步，每 100 步按同类验证目标选 checkpoint。
验证最优步分别为基线 1100、A1 900。`policy-last.pt` 保留末步，`policy.pt`
为选出的版本。早期的 300 步和仅向量约束版本单独保留，不能混作最终结果。

### 双点 A 参考与限制

`planning/endpoint_flow.py` 在当前前方扇面内搜索中心和方向，构造跨度严格
等于控制器输入值的对称双点；示例初始跨度 0.82m。只接受中点场方向与连线
夹角 85–95°、端点及连接线满足 2cm 栅格净空的候选。中心搜索步长 0.1m，
方向步长 5°、相对视轴 ±40°，因此失败不代表穷尽搜索后的不可行证明。

两端点用相同积分时间与同一速度缩放推进，dt=0.02s，前瞻最多 0.6s，
每 0.2s 更新。参考 COG 为中点，跨度为端点距离，yaw 为连线法向；这只适用
于对称横向端点，不能自动替代真实机构运动学。示例限制：最小跨度 0.60m、
跨度变化率 0.25m/s、yaw 速率 0.8rad/s、端点速度 0.5m/s，且默认只收缩。
这些是实验配置，0.60m 仅参考旧硬件资料的近似外宽，均不是现机标定。

A 保留上周期世界坐标参考和 1s 有效期场缓存；新旧场方向差小于 30°才平均，
否则以当前观测更新，不平均矛盾分支。候选只在旧参考约 2cm/5° 邻域内接续，
用保留起点和首步速度的五次权重过渡修正后续参考，并重新检查速率、净空、
垂直条件。没有可行接续就返回失败，旧参考过期后要求显式 reset，不静默
跳到另一组。B 使用相同几何与速率限制，但每帧独立选点。

本次严格要求每次更新的初始组合仍位于当前扇面。场缓存不能自动豁免此条件；
离开当前扇面的历史组合会被拒绝。没有实现“最小跨度到达后偏离原流线保持
宽度”或“放宽当前扇面限制”的策略，避免在验证时偷偷改动用户条件。

### 重现命令与产物

从采集到训练、基线比较和重放的一条命令（使用新的空目录）：

```bash
python -m cofl.experiments.cofm_temporal \
  --config artifacts/cof-m/matterport-point-020/studio.yaml \
  --tasks configs/experiments/cofm_temporal_tasks.json \
  --output artifacts/cof-m/temporal-a-new \
  --max-frames 24 --steps 1200 --device cuda \
  --width 0.82 --min-width 0.60
```

可用 `--stage collect` 单独采集；随后相同 output 配 `--stage train` 训练。
重新评估已有产物，不训练：

```bash
python -m cofl.experiments.cofm_temporal \
  --stage evaluate --output artifacts/cof-m/temporal-a-20261122 --device cuda
```

`--width` 与 `--min-width` 可改变跨度；`--allow-expansion` 可关闭单调收缩
限制。修改这些参数会改变实验条件，应使用新的产物目录或保留原 report。
产物包括 `report.json`、`baseline/`、`temporal/`、`dataset/`、逐帧
`teacher_A/trained_A/teacher_B/trained_B-references.json`、`continuity.gif`、
`continuity-metrics.png`、`preview.png` 与 `feasible-preview.png`。

### 本次实测结论（2026-11-22）

| 验证指标 | 单帧监督基线 | 加跨帧约束 |
| --- | --- | --- |
| 跨帧世界向量 RMSE | 0.122651 | 0.069351 |
| 跨帧方向差均值 | 11.80° | 7.13° |
| 跨帧方向差 P95 | 36.39° | 20.68° |
| 单帧方向相对 teacher 误差 | 24.55° | 25.08° |
| 单帧向量 RMSE | 0.228251 | 0.236489 |

teacher 在相同世界点上的跨帧 RMSE 约 1.38e-8。跨帧训练确实减少抖动，
但单帧准确率没有改善，局部仍可能出现较大方向跳变；不能把“更稳定”当作
“已足够准确”。严格 A 重放中，teacher 可生成 5/48 帧可行参考，网络为
3/48；通过的接续起点、yaw 和首步速度误差均为零，但未全程通过。
B 的 teacher/网络分别为 25/48、36/48，B 会重新选择，不能据帧数多认定
实际飞行更可靠，也不能与 A 的连续跟踪成功率混为一谈。

失败分开记录：65cm 任务的 teacher 会到达最小跨度后仍要求继续收缩；曲线
任务的旧组合会离开当前扇面；网络额外存在垂直角超限、接续或速率限制。
`reference_expired` 是此前失败后剩余参考耗尽，不是另一种网络误差。

重放采用记录的 COG/相机位姿与理想跨度反馈，不是模拟动力学，也没有让
飞行器沿网络规划后的 COG 路径重新渲染闭环观测。净空检查还使用真实地图
作为离线 oracle，只检查双点及连接线，不验证另两条臂、真实单自由度耦合
或飞控。前方截面与实际 COG 的接入尚未接通；点路径及参考生成成功不能
证明 82cm 机体已穿过 65cm 窄口。当前可复现结论是 teacher 一致性成立、
跨帧训练有效、严格双点规则还不能全程运行。

验证本次几何、帧对齐、记忆有效期、接续、二维 checkpoint 和回归：

```bash
python -m pytest -q reproduce/test_cofm_temporal.py \
  reproduce/test_matterport_routes.py reproduce/test_point_field.py \
  reproduce/test_cofm_studio.py reproduce/test_configuration_field.py
```


## 8. 约 8 小时的 A1 续训（2026-11-22 启动）

入口为 `python -m cofl.experiments.cofm_long_train`。一个 epoch 定义为将
28 个训练帧对随机打乱后完整遍历一次，不放回；每次访问仍重新采样 512 个
查询。此前短实验按 step 随机抽帧对，不能直接把一个 step 叫作 epoch。

RTX 5080 实测 6 个 epoch，首轮预热后每 epoch 中位约 0.629s，验证另约
0.111s。预计约 45,697 epoch 对应 8h，因此设置 46,000 epoch 与累计运行
时间 8h 双重上限，先到者停止训练。恢复进程时累计此前已运行时间，停机
期间不计入训练时间。基准记录在 `artifacts/cof-m/temporal-a-8h-20261122/benchmark/benchmark.json`。

本次继续使用同一 48 帧数据、38/10 帧划分和 teacher，不扩充或改动验证集。
从此前 `temporal/policy.pt` 验证最优模型继续，所有 encoder/decoder 都训练。
旧权重没有优化器状态，因此首次续训新建 AdamW，lr=1e-4、weight_decay=1e-4，
跨帧权重 2、梯度裁剪 1；验证不改善 10 次后学习率乘 0.5，最低 1e-6。
不因验证停滞提前结束 8h，但始终保留验证最优模型，避免把末步当最优。

后台输出在 `artifacts/cof-m/temporal-a-8h-20261122/`：

- `launcher.json`：启动 PID、命令和预计结束时间；`console.log`：训练输出。
- `run/status.json`：当前 epoch、step、运行时间、学习率及最佳验证分数。
- `run/policy.pt`：验证最优模型；`policy-last.pt`：最近保存的当前模型。
- `run/resume.pt`：模型、AdamW、学习率调度器、随机状态、epoch 内游标及累计时间。
- `run/validation.jsonl`：验证历史；每 95 epoch（约 1min）验证一次。
- `run/metrics-latest.json`、`metrics-history.jsonl`：每约 30min 对最优模型做完整流场评估。
- 结束后 `run/evaluation-final.json`：完整流场指标和严格 A/B 重放；同时输出 GIF 与曲线。

每 5min 自动原子保存断点，已实际测试 CUDA 上从第 6 epoch/168 step 恢复
至第 7 epoch/196 step。SIGTERM/SIGINT 会请求在当前更新结束后保存退出。
不重复启动同一 output，进程锁会拒绝并发运行。重启电脑不会自动恢复。

中断后恢复（原进程已停止时）：

```bash
python -m cofl.experiments.cofm_long_train \
  --resume --epochs 46000 --hours 8 --validate-every 95 \
  --output artifacts/cof-m/temporal-a-8h-20261122/run
```

读取状态：

```bash
python -c "import json; print(json.dumps(json.load(open('artifacts/cof-m/temporal-a-8h-20261122/run/status.json')), indent=2))"
```

同一小场景长时间重复训练可能过拟合。需同时看 teacher 方向误差、跨帧差、
双点接续和失败原因，不能只凭训练 loss 降低认定能穿缝。长训不会自动解除
上一节发现的最小跨度、当前扇面限制或真实 COG 接入问题。
