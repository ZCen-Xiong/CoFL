import type { ComponentType } from "react";
import type { ModeProps } from "./types";
import { ScenePlayground } from "./ScenePlayground";
import { ValInspector } from "./ValInspector";
import { OnlinePlayground } from "./OnlinePlayground";
import { CoFLMPlayground } from "./CoFLMPlayground";

/** Add a renderer here; backend modes independently own their execution protocol. */
export const modeRenderers: Readonly<Record<string, ComponentType<ModeProps>>> =
  {
    coflm: CoFLMPlayground,
    scene: ScenePlayground,
    validation: ValInspector,
    online: OnlinePlayground,
  };
