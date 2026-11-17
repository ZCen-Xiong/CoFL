import { useEffect, useRef, useState } from "react";
import { Badge, Button, Callout, Flex, Select, Text } from "@radix-ui/themes";
import { api, apiError } from "../api/client";
import type { components } from "../api/schema";

type Scene = components["schemas"]["MScene"];
type Dock = components["schemas"]["MDock"];
type Teacher = components["schemas"]["MTeacher"];
type Observation = components["schemas"]["MObservation"];
type Summary = components["schemas"]["MSceneSummary"];
type XY = [number, number];

function Cross({
  point,
  color,
  allShapes = false,
  q = 0,
}: {
  point: number[];
  color: string;
  allShapes?: boolean;
  q?: number;
}) {
  const x = point[0],
    y = point[1];
  const ex = allShapes ? 0.4 : 0.3 + q;
  const ey = allShapes ? 0.4 : 0.3 - q;
  return (
    <g stroke={color} strokeLinecap="round" opacity={allShapes ? 0.65 : 1}>
      <line x1={x - ex} x2={x + ex} y1={y} y2={y} strokeWidth={0.2} />
      <line x1={x} x2={x} y1={y - ey} y2={y + ey} strokeWidth={0.2} />
      <circle cx={x} cy={y} r={0.10} fill={color} stroke="none" />
    </g>
  );
}

export function CofMPlayground() {
  const [scenes, setScenes] = useState<Summary[]>([]);
  const [sceneId, setSceneId] = useState("");
  const [scene, setScene] = useState<Scene | null>(null);
  const [start, setStart] = useState<Dock | null>(null);
  const [goal, setGoal] = useState<Dock | null>(null);
  const [target, setTarget] = useState<"start" | "goal">("start");
  const [teacher, setTeacher] = useState<Teacher | null>(null);
  const [observation, setObservation] = useState<Observation | null>(null);
  const [error, setError] = useState("");
  const [depthError, setDepthError] = useState("");
  const [busy, setBusy] = useState(false);
  const [slice, setSlice] = useState(4);
  const [cursor, setCursor] = useState(0);
  const revision = useRef(0);
  const seed = useRef(0);

  useEffect(() => {
    let active = true;
    void api.GET("/api/v1/cofm/scenes").then(({ data, error }) => {
      if (!active) return;
      if (error || !data) {
        setError(apiError(error).message);
        return;
      }
      setScenes(data);
      setSceneId(data[0]?.id ?? "");
    });
    return () => {
      active = false;
      revision.current++;
    };
  }, []);

  async function renderDepth(
    id: string,
    point: number[],
    q: number,
    version: number,
  ) {
    try {
      const { data, error } = await api.POST("/api/v1/cofm/observation", {
        body: { scene_id: id, point: point as XY, q_m: q },
      });
      if (error || !data) throw apiError(error);
      if (version === revision.current) {
        setObservation(data as Observation);
        setDepthError("");
      }
    } catch (e) {
      if (version === revision.current) setDepthError((e as Error).message);
    }
  }

  async function calculate(id: string, a: Dock, b: Dock, version: number) {
    if (!a.valid || !b.valid) return;
    await Promise.all([
      (async () => {
        try {
          const { data, error } = await api.POST("/api/v1/cofm/teacher", {
            body: { scene_id: id, start: a.point, goal: b.point },
          });
          if (error || !data) throw apiError(error);
          if (version === revision.current) {
            setTeacher(data as Teacher);
            setCursor(0);
            setSlice(4);
          }
        } catch (e) {
          if (version === revision.current) setError((e as Error).message);
        }
      })(),
      renderDepth(id, a.point, 0, version),
    ]);
  }

  async function randomize(id = sceneId, version = ++revision.current) {
    setBusy(true);
    setError("");
    setDepthError("");
    setTeacher(null);
    setObservation(null);
    try {
      const { data, error } = await api.POST("/api/v1/cofm/sample", {
        body: { scene_id: id, seed: seed.current++ },
      });
      if (error || !data) throw apiError(error);
      if (version !== revision.current) return;
      setStart(data.start as Dock);
      setGoal(data.goal as Dock);
      await calculate(id, data.start as Dock, data.goal as Dock, version);
    } catch (e) {
      if (version === revision.current) setError((e as Error).message);
    } finally {
      if (version === revision.current) setBusy(false);
    }
  }

  useEffect(() => {
    if (!sceneId) return;
    const version = ++revision.current;
    setScene(null);
    setStart(null);
    setGoal(null);
    setTeacher(null);
    setObservation(null);
    setBusy(true);
    setError("");
    setDepthError("");
    void (async () => {
      try {
        const { data, error } = await api.GET(
          "/api/v1/cofm/scenes/{scene_id}",
          {
            params: { path: { scene_id: sceneId } },
          },
        );
        if (error || !data) throw apiError(error);
        if (version !== revision.current) return;
        setScene(data as Scene);
        await randomize(sceneId, version);
      } catch (e) {
        if (version === revision.current) {
          setError((e as Error).message);
          setBusy(false);
        }
      }
    })();
    // Scene changes own the lifecycle; late responses cannot populate another scene.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneId]);

  async function choose(point: XY) {
    const version = ++revision.current;
    setBusy(true);
    setTeacher(null);
    setObservation(null);
    setError("");
    setDepthError("");
    try {
      const { data, error } = await api.POST("/api/v1/cofm/validate", {
        body: { scene_id: sceneId, point },
      });
      if (error || !data) throw apiError(error);
      if (version !== revision.current) return;
      if (target === "start") setStart(data as Dock);
      else setGoal(data as Dock);
      if (!data.valid) setError(data.reason);
      else if (target === "start")
        await renderDepth(sceneId, data.point, 0, version);
    } catch (e) {
      if (version === revision.current) setError((e as Error).message);
    } finally {
      if (version === revision.current) setBusy(false);
    }
  }

  const selected = teacher?.path[cursor] ?? [...(start?.point ?? [0, 0]), 0];
  const field = teacher?.slices[slice]?.arrows ?? [];
  const qSlice = teacher?.slices[slice]?.q_m ?? 0;
  const local = field.filter(([x, y]) => {
    const dx = x - selected[0],
      dy = y - selected[1];
    return dx >= 0 && Math.abs(dy) <= dx && Math.hypot(dx, dy) <= 5;
  });
  const staleDepth =
    observation &&
    (Math.hypot(
      observation.position[0] - selected[0],
      observation.position[1] - selected[1],
    ) > 0.01 ||
      Math.abs(observation.q_m - selected[2]) > 0.001);

  return (
    <main className="cofm-workspace">
      <aside className="cofm-controls">
        <div className="control-section">
          <div className="section-label">cof-m · DEPTH ONLY</div>
          <Select.Root value={sceneId} onValueChange={setSceneId}>
            <Select.Trigger
              aria-label="cof-m scene"
              placeholder="选择场景"
              className="full-width"
            />
            <Select.Content>
              {scenes.map((s) => (
                <Select.Item key={s.id} value={s.id}>
                  {s.label}
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
          {!scenes.length && (
            <Text as="p" size="2">
              请使用含 cof-m 场景的 studio.yaml 启动。
            </Text>
          )}
          <Text as="p" size="2">
            固定高度 {scene?.height_m ?? 1.2}m · XOY 平面 · 朝向 +X
          </Text>
          <Badge color="teal">几何 teacher · 未使用训练权重</Badge>
        </div>
        <div className="control-section">
          <div className="section-label">起点与停泊目标</div>
          <Button disabled={busy || !scene} onClick={() => void randomize()}>
            随机生成可达两点
          </Button>
          <Flex gap="2" mt="3">
            <Button
              variant={target === "start" ? "solid" : "soft"}
              onClick={() => setTarget("start")}
            >
              地图选起点
            </Button>
            <Button
              variant={target === "goal" ? "solid" : "soft"}
              onClick={() => setTarget("goal")}
            >
              地图选终点
            </Button>
          </Flex>
          <Text as="p" size="2">
            点击下方选点地图，对齐 0.1m 网格并校验。
          </Text>
          {(
            [
              ["起点", start],
              ["终点", goal],
            ] as const
          ).map(([label, dock]) => (
            <div className="cofm-dock" key={label}>
              <strong>
                {label}{" "}
                {dock
                  ? `(${dock.point.map((n) => n.toFixed(1)).join(", ")})m`
                  : "未选择"}
              </strong>
              <Badge color={dock?.valid ? "green" : "red"}>
                {dock?.valid ? "全形态停泊通过" : "未通过"}
              </Badge>
              {dock && (
                <small>最小净空 {(dock.clearance_m * 100).toFixed(1)}cm</small>
              )}
            </div>
          ))}
          <Button
            mt="3"
            disabled={busy || !start?.valid || !goal?.valid}
            onClick={() => {
              const version = ++revision.current;
              setBusy(true);
              setError("");
              setTeacher(null);
              setObservation(null);
              void calculate(sceneId, start!, goal!, version).finally(() => {
                if (version === revision.current) setBusy(false);
              });
            }}
          >
            生成 teacher 与第一人称深度
          </Button>
        </div>
        <div className="control-section">
          <div className="section-label">停泊包络</div>
          <Text as="p" size="2">
            全部 q∈[−0.10,+0.10]m 的十字包络：
          </Text>
          <Text as="p" size="2">
            初始横纵外轮廓各长 80cm，面宽恒定 20cm。横长 80+200q cm，纵长
            80−200q cm（q 单位为米），各在 60–100cm 内对称伸缩，总长恒为 160cm。
            停泊位置检查全部形态的并集：横纵各跨 100cm，另留 1cm 裕度。
          </Text>
          <Text as="p" size="2">
            检查整块胶囊十字与障碍及规划边界。保证代理模型的全部伸缩形态，不包含任意旋转姿态。
          </Text>
        </div>
      </aside>
      <section className="cofm-main">
        {busy && (
          <Callout.Root>
            <Callout.Text>
              正在检查停泊、求解 teacher 或渲染 Habitat 深度…
            </Callout.Text>
          </Callout.Root>
        )}
        {error && (
          <Callout.Root color="red" role="alert">
            <Callout.Text>{error}</Callout.Text>
          </Callout.Root>
        )}
        <div className="cofm-observations">
          <section className="cofm-panel">
            <div className="section-label">第一人称 · HABITAT DEPTH</div>
            {observation ? (
              <img
                className="cofm-depth"
                src={observation.image}
                alt="Habitat first-person depth at fixed flight height"
              />
            ) : (
              <div className="cofm-placeholder">
                选择有效起点后采集单通道深度
              </div>
            )}
            {observation && (
              <Text as="p" size="2">
                相机 ({observation.position.map((n) => n.toFixed(2)).join(", ")}
                )m，q={observation.q_m.toFixed(3)}m；有效深度{" "}
                {(observation.valid_fraction * 100).toFixed(1)}
                %。亮=近，暗=远，黑=无效。
              </Text>
            )}
            {depthError && (
              <Callout.Root color="red">
                <Callout.Text>{depthError}</Callout.Text>
              </Callout.Root>
            )}
            {staleDepth && (
              <Text as="p" color="amber" size="2">
                已更改路径构型；点击下方按钮更新深度。
              </Text>
            )}
            <Button
              size="2"
              variant="soft"
              disabled={busy || !start?.valid}
              onClick={() => {
                const version = ++revision.current;
                setBusy(true);
                void renderDepth(
                  sceneId,
                  selected.slice(0, 2),
                  selected[2],
                  version,
                ).finally(() => {
                  if (version === revision.current) setBusy(false);
                });
              }}
            >
              更新当前构型深度
            </Button>
          </section>
          <section className="cofm-panel">
            <div className="section-label">局部流场 · 前方 5m / HFOV 90°</div>
            <svg
              className="cofm-sector"
              viewBox="-5 -5.2 10 5.7"
              aria-label="cof-m local teacher field"
            >
              <path
                d="M0 0 L-3.535 -3.535 A5 5 0 0 1 3.535 -3.535 Z"
                fill="#eaf1ec"
                stroke="#a5bcb1"
                strokeWidth=".025"
              />
              {local.map(([x, y, fx, fy, fq], i) =>
                Math.hypot(fx, fy) > 0 ? (
                  <line
                    key={i}
                    x1={-(y - selected[1])}
                    y1={-(x - selected[0])}
                    x2={-(y - selected[1]) - 0.25 * fy}
                    y2={-(x - selected[0]) - 0.25 * fx}
                    stroke="#277f72"
                    strokeWidth=".035"
                    markerEnd="url(#cofm-local-arrow)"
                  />
                ) : (
                  fq !== 0 && (
                    <circle
                      key={i}
                      cx={-(y - selected[1])}
                      cy={-(x - selected[0])}
                      r=".045"
                      fill="#c47b24"
                    />
                  )
                ),
              )}
              <defs>
                <marker
                  id="cofm-local-arrow"
                  markerWidth="4"
                  markerHeight="4"
                  refX="3"
                  refY="2"
                  orient="auto"
                >
                  <path d="M0 0 L4 2 L0 4" fill="#277f72" />
                </marker>
              </defs>
              <circle cx="0" cy="0" r=".1" fill="#25463d" />
              <text
                x="0"
                y="-4.85"
                textAnchor="middle"
                fontSize=".25"
                fill="#526b60"
              >
                前方 +X
              </text>
            </svg>
            <Text as="p" size="2">
              以当前构型为原点的 XOY 局部图，不是地面投影。橙点=原地变形；q
              切片={qSlice.toFixed(3)}m。
            </Text>
            <label>
              显示 q 切片{" "}
              <input
                aria-label="Teacher q slice"
                type="range"
                min="0"
                max="8"
                value={slice}
                onChange={(e) => setSlice(Number(e.target.value))}
              />
            </label>
          </section>
        </div>
        <section className="cofm-panel">
          <Flex align="center" justify="between" wrap="wrap" gap="2">
            <div className="section-label">
              XOY 选点与路径地图 · 正在选择
              {target === "start" ? "起点" : "终点"}
            </div>
            {teacher && (
              <Badge color={teacher.reachable ? "green" : "red"}>
                {teacher.reachable ? "teacher 可达" : "图中不可达"}
              </Badge>
            )}
          </Flex>
          {scene && (
            <svg
              className="cofm-map"
              viewBox={`0 0 ${scene.length_m} ${scene.width_m}`}
              aria-label="cof-m docking map"
              onClick={(e) => {
                if (busy) return;
                const svg = e.currentTarget,
                  matrix = svg.getScreenCTM();
                if (!matrix) return;
                const p = svg.createSVGPoint();
                p.x = e.clientX;
                p.y = e.clientY;
                const xy = p.matrixTransform(matrix.inverse());
                void choose([
                  Math.round(xy.x * 10) / 10,
                  Math.round((scene.width_m / 2 - xy.y) * 10) / 10,
                ]);
              }}
            >
              <rect
                width={scene.length_m}
                height={scene.width_m}
                fill="#f1f4ef"
              />
              {scene.obstacles.map((o, i) =>
                o.kind === "circle" ? (
                  <circle
                    key={i}
                    cx={o.center[0]}
                    cy={scene.width_m / 2 - o.center[1]}
                    r={o.radius}
                    fill="#70877d"
                  />
                ) : (
                  <rect
                    key={i}
                    x={o.center[0] - o.size![0] / 2}
                    y={scene.width_m / 2 - o.center[1] - o.size![1] / 2}
                    width={o.size![0]}
                    height={o.size![1]}
                    fill="#7c817a"
                  />
                ),
              )}
              {teacher?.path.length ? (
                <polyline
                  points={teacher.path
                    .map(([x, y]) => `${x},${scene.width_m / 2 - y}`)
                    .join(" ")}
                  stroke="#2d7d9a"
                  strokeWidth=".065"
                  fill="none"
                />
              ) : null}
              {field.map(
                ([x, y, fx, fy], i) =>
                  Math.hypot(fx, fy) > 0 && (
                    <line
                      key={i}
                      x1={x}
                      y1={scene.width_m / 2 - y}
                      x2={x + 0.2 * fx}
                      y2={scene.width_m / 2 - y - 0.2 * fy}
                      stroke="#90b4a3"
                      strokeWidth=".025"
                    />
                  ),
              )}
              {(
                [
                  ["S", start, "#268074"],
                  ["G", goal, "#b97a27"],
                ] as const
              ).map(
                ([label, dock, color]) =>
                  dock && (
                    <g key={label}>
                      <Cross
                        point={[
                          dock.point[0],
                          scene.width_m / 2 - dock.point[1],
                        ]}
                        color={dock.valid ? color : "#d33c39"}
                        allShapes
                      />
                      <text
                        x={dock.point[0]}
                        y={scene.width_m / 2 - dock.point[1] + 0.1}
                        textAnchor="middle"
                        fontSize=".27"
                        fill="white"
                      >
                        {label}
                      </text>
                    </g>
                  ),
              )}
              {teacher?.path.length ? (
                <Cross
                  point={[selected[0], scene.width_m / 2 - selected[1]]}
                  q={selected[2]}
                  color="#244e70"
                />
              ) : null}
            </svg>
          )}
          <Text as="p" size="2">
            横轴 x：0–{scene?.length_m}m；纵轴 y：−{(scene?.width_m ?? 10) / 2}{" "}
            至 +{(scene?.width_m ?? 10) / 2}m。S/G
            十字显示全形态停泊包络。所有标签位于 z={scene?.height_m ?? 1.2}m。
          </Text>
          {teacher && (
            <>
              <Text as="p" size="2">
                {teacher.message}。路径 {teacher.path.length} 个构型。
                {teacher.min_path_clearance_m != null &&
                  `扫掠净空下界 ${(teacher.min_path_clearance_m * 100).toFixed(1)}cm。`}
              </Text>
              {teacher.path.length > 0 && (
                <label>
                  浏览 teacher 路径：{cursor + 1}/{teacher.path.length}，q=
                  {selected[2].toFixed(3)}m
                  <input
                    aria-label="Teacher path position"
                    type="range"
                    min="0"
                    max={teacher.path.length - 1}
                    value={cursor}
                    onChange={(e) => {
                      const i = Number(e.target.value);
                      setCursor(i);
                      setSlice(Math.round((teacher.path[i][2] + 0.1) / 0.025));
                    }}
                  />
                </label>
              )}
              <Text as="p" size="1">
                Teacher 已保存：studio-tasks/{teacher.artifact_id}
                /teacher.npz。这是完整几何生成的全局监督，非 DFormer
                预测；不代表单帧深度足以推断遮挡后的路径。
              </Text>
              <a
                href={`/api/v1/cofm/tasks/${teacher.artifact_id}/teacher`}
                download
              >
                下载当前 teacher.npz
              </a>
            </>
          )}
        </section>
      </section>
    </main>
  );
}
