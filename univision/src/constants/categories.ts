const DEFAULT_FALLBACK_COLOR = "#94a3b8";

export const CATEGORY_COLORS: Record<string, string> = {
  Input: "#22d3ee",
  Ingestion: "#06b6d4",
  Detection: "#f43f5e",
  Preprocessing: "#fb923c",
  OCR: "#4ade80",
  PostProcessing: "#facc15",
  Output: "#60a5fa",
  Utility: "#94a3b8",
};

/** Merge backend categories into the colour map. */
export function mergeCategoryColors(backend: Record<string, string>) {
  Object.assign(CATEGORY_COLORS, backend);
}

/** Get colour for category with fallback. */
export function getCategoryColor(category: string): string {
  return CATEGORY_COLORS[category] ?? DEFAULT_FALLBACK_COLOR;
}
