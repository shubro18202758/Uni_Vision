const DEFAULT_FALLBACK_COLOR = "#94a3b8";

export const PORT_COLORS: Record<string, string> = {
  frame: "#60a5fa",
  bounding_box_list: "#fb923c",
  text: "#4ade80",
  config: "#f8fafc",
  number: "#7dd3fc",
  boolean: "#c084fc",
  any: "#e2e8f0",
};

/** Merge backend port types into the colour map. */
export function mergePortColors(backend: Record<string, string>) {
  Object.assign(PORT_COLORS, backend);
}

/** Get colour for port type with fallback. */
export function getPortColor(type: string): string {
  return PORT_COLORS[type] ?? DEFAULT_FALLBACK_COLOR;
}
