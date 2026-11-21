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

function Point({ point, color }: { point: number[]; color: string }) {
  return <circle cx={point[0]} cy={point[1]} r={0.06} fill={color} />;
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
  const [narrowOnly, setNarrowOnly] = useState<boolean | null>(null);
  const [cameraView, setCameraView] = useState<"rgb" | "depth">("rgb");
  const [busy, setBusy] = useState(false);
  const [cursor, setCursor] = useState(0);
  const revision = useRef(0);

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

  async function renderObservation(
    id: string,
    point: number[],
    version: number,
  ) {
    try {
      const { data, error } = await api.POST("/api/v1/cofm/observation", {
        body: { scene_id: id, point: point as XY },
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

  async function calculate(
    id: string,
    a: Dock,
    b: Dock,
    version: number,
    narrow = narrowOnly,
  ) {
    if (!a.valid || !b.valid) return;
    await Promise.all([
      (async () => {
        try {
          const { data, error } = await api.POST("/api/v1/cofm/teacher", {
            body: {
              scene_id: id,
              start: a.point,
              goal: b.point,
              narrow_only: narrow,
            },
          });
          if (error || !data) throw apiError(error);
          if (version === revision.current) {
            setTeacher(data as Teacher);
            setCursor(0);
          }
        } catch (e) {
          if (version === revision.current) setError((e as Error).message);
        }
      })(),
      renderObservation(id, a.point, version),
    ]);
  }

  useEffect(() => {
    if (!sceneId) return;
    const version = ++revision.current;
    setScene(null);
    setNarrowOnly(null);
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
        setBusy(false);
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
        await renderObservation(sceneId, data.point, version);
    } catch (e) {
      if (version === revision.current) setError((e as Error).message);
    } finally {
      if (version === revision.current) setBusy(false);
    }
  }

  const selected = teacher?.path[cursor] ?? start?.point ?? [0, 0];
  const field = teacher?.arrows ?? [];
  const local = field.filter(([x, y]) => {
    const dx = x - selected[0],
      dy = y - selected[1];
    return dx >= 0 && Math.abs(dy) <= dx && Math.hypot(dx, dy) <= 5;
  });
  const staleDepth =
    observation &&
    Math.hypot(
      observation.position[0] - selected[0],
      observation.position[1] - selected[1],
    ) > 0.01;

  return (
    <main className="cofm-workspace">
      <aside className="cofm-controls">
        <div className="control-section">
          <div className="section-label">cof-m · RGB-D · 无语言</div>
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
          {scene?.source_kind === "matterport" && (
            <Text as="p" size="2">
              Matterport 地面占据近似 · 固定高度平飞
            </Text>
          )}
        </div>
        <div className="control-section">
          <div className="section-label">手动起点与终点</div>
          {scene?.source_kind === "matterport" && (
            <label>
              <input
                type="checkbox"
                checked={narrowOnly ?? false}
                disabled={busy}
                onChange={(e) => {
                  setNarrowOnly(e.target.checked);
                  setTeacher(null);
                }}
              />
              仅接受包含窄段的路径（宽≤1.1m，连续≥0.3m）
            </label>
          )}
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
            点击下方地图，按场景分辨率对齐并检查质点净空。
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
                  ? `(${dock.point.map((n) => n.toFixed(3)).join(", ")})m`
                  : "未选择"}
              </strong>
              <Badge color={dock?.valid ? "green" : "red"}>
                {dock?.valid ? "质点净空通过" : "未通过"}
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
            生成 teacher 与第一人称 RGB-D
          </Button>
        </div>
        <div className="control-section">
          <div className="section-label">质点向量场 F(x,y)</div>
          <Text as="p" size="2">
            墙面净空 {Number(scene?.docking.margin_m ?? 0.02) * 100}cm。
            不按十字整机膨胀障碍，不使用 q 切片；COG 附近流线由后续阶段提取。
          </Text>
        </div>
      </aside>
      <section className="cofm-main">
        {busy && (
          <Callout.Root>
            <Callout.Text>
              正在检查质点净空、求解 teacher 或渲染 Habitat RGB-D…
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
            <div className="section-label">第一人称 · HABITAT RGB-D</div>
            <Flex gap="2" mb="3">
              <Button
                size="1"
                variant={cameraView === "rgb" ? "solid" : "soft"}
                onClick={() => setCameraView("rgb")}
              >
                RGB
              </Button>
              <Button
                size="1"
                variant={cameraView === "depth" ? "solid" : "soft"}
                onClick={() => setCameraView("depth")}
              >
                深度
              </Button>
            </Flex>
            {observation ? (
              <img
                className="cofm-depth"
                src={
                  cameraView === "rgb" && observation.rgb_image
                    ? observation.rgb_image
                    : observation.image
                }
                alt={`Habitat first-person ${cameraView} at fixed flight height`}
              />
            ) : (
              <div className="cofm-placeholder">
                选择有效起点后采集对齐的 RGB-D
              </div>
            )}
            {observation && (
              <Text as="p" size="2">
                相机 ({observation.position.map((n) => n.toFixed(2)).join(", ")}
                )m；有效深度 {(observation.valid_fraction * 100).toFixed(1)}
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
                已更改路径位置；点击下方按钮更新 RGB-D。
              </Text>
            )}
            <Button
              size="2"
              variant="soft"
              disabled={busy || !start?.valid}
              onClick={() => {
                const version = ++revision.current;
                setBusy(true);
                void renderObservation(
                  sceneId,
                  selected.slice(0, 2),
                  version,
                ).finally(() => {
                  if (version === revision.current) setBusy(false);
                });
              }}
            >
              更新当前位置 RGB-D
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
              {local.map(([x, y, fx, fy], i) =>
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
                ) : null,
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
              以当前位置为原点的质点二维场，复用 CoFL-S 测地势场算法，无 q
              分量。
            </Text>
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
                  Math.round(xy.x / scene.resolution_m) * scene.resolution_m,
                  Math.round((scene.width_m / 2 - xy.y) / scene.resolution_m) *
                    scene.resolution_m,
                ]);
              }}
            >
              <rect
                width={scene.length_m}
                height={scene.width_m}
                fill="#f1f4ef"
              />
              {scene.map_image && (
                <image
                  href={scene.map_image}
                  x="0"
                  y="0"
                  width={scene.length_m}
                  height={scene.width_m}
                  preserveAspectRatio="none"
                />
              )}
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
                      <Point
                        point={[
                          dock.point[0],
                          scene.width_m / 2 - dock.point[1],
                        ]}
                        color={dock.valid ? color : "#d33c39"}
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
                <Point
                  point={[selected[0], scene.width_m / 2 - selected[1]]}
                  color="#244e70"
                />
              ) : null}
            </svg>
          )}
          <Text as="p" size="2">
            横轴 x：0–{scene?.length_m}m；纵轴 y：−{(scene?.width_m ?? 10) / 2}{" "}
            至 +{(scene?.width_m ?? 10) / 2}m。S/G 为质点端点。所有标签位于 z=
            {scene?.height_m ?? 1.2}m。
          </Text>
          {teacher && (
            <>
              <Text as="p" size="2">
                {teacher.message}。路径 {teacher.path.length} 个位置。
                {teacher.narrow &&
                  `窄段 ${Number(teacher.narrow.narrow_length_m).toFixed(2)}m；窄段附近可采集 ${(teacher.narrow.selected_indices as number[]).length} 帧。`}
                {teacher.min_path_clearance_m != null &&
                  `可达网格最小净空 ${(teacher.min_path_clearance_m * 100).toFixed(1)}cm。`}
              </Text>
              {teacher.path.length > 0 && (
                <label>
                  浏览质点路径：{cursor + 1}/{teacher.path.length}
                  <input
                    aria-label="Teacher path position"
                    type="range"
                    min="0"
                    max={teacher.path.length - 1}
                    value={cursor}
                    onChange={(e) => {
                      const i = Number(e.target.value);
                      setCursor(i);
                    }}
                  />
                </label>
              )}
              <Text as="p" size="1">
                Teacher 已保存：studio-tasks/{teacher.artifact_id}
                /teacher.npz。这是完整几何生成的全局监督，非 DFormer 预测；单帧
                RGB-D 无法保证推断遮挡后的路径。
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
