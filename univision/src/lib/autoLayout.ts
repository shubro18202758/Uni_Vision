import dagre from "dagre";
import type { Edge, Node } from "reactflow";

/* ------------------------------------------------------------------ */
/*  Layout algorithm types                                            */
/* ------------------------------------------------------------------ */

export type LayoutAlgorithm =
  | "mindmap"
  | "force"
  | "radial"
  | "hierarchical-lr"
  | "hierarchical-tb"
  | "grid"
  | "manual";

export interface LayoutOptions {
  nodeWidth?: number;
  nodeHeight?: number;
  spacing?: number;
  iterations?: number;
  centerX?: number;
  centerY?: number;
}

const DEFAULTS: Required<LayoutOptions> = {
  nodeWidth: 260,
  nodeHeight: 160,
  spacing: 60,
  iterations: 300,
  centerX: 600,
  centerY: 400,
};

/* ------------------------------------------------------------------ */
/*  Force-directed (spring-charge simulation)                         */
/* ------------------------------------------------------------------ */

function forceLayout(nodes: Node[], edges: Edge[], o: Required<LayoutOptions>): Node[] {
  if (nodes.length === 0) return [];

  const pos = new Map<string, { x: number; y: number; vx: number; vy: number }>();
  nodes.forEach((n, i) => {
    const a = (2 * Math.PI * i) / nodes.length;
    const r = Math.max(200, nodes.length * 40);
    pos.set(n.id, { x: o.centerX + r * Math.cos(a), y: o.centerY + r * Math.sin(a), vx: 0, vy: 0 });
  });

  const springLen = o.spacing + o.nodeWidth;
  const repulsion = 8000;
  const springK = 0.01;
  const damping = 0.85;

  for (let t = 0; t < o.iterations; t++) {
    const alpha = 1 - t / o.iterations;

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = pos.get(nodes[i].id)!;
        const b = pos.get(nodes[j].id)!;
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        const d = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        const f = (repulsion * alpha) / (d * d);
        dx = (dx / d) * f;
        dy = (dy / d) * f;
        a.vx -= dx; a.vy -= dy;
        b.vx += dx; b.vy += dy;
      }
    }

    for (const e of edges) {
      const a = pos.get(e.source);
      const b = pos.get(e.target);
      if (!a || !b) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const f = springK * (d - springLen) * alpha;
      const fx = (dx / d) * f;
      const fy = (dy / d) * f;
      a.vx += fx; a.vy += fy;
      b.vx -= fx; b.vy -= fy;
    }

    pos.forEach((p) => {
      p.vx += (o.centerX - p.x) * 0.001 * alpha;
      p.vy += (o.centerY - p.y) * 0.001 * alpha;
      p.vx *= damping; p.vy *= damping;
      p.x += p.vx; p.y += p.vy;
    });
  }

  return nodes.map((n) => {
    const p = pos.get(n.id)!;
    return { ...n, position: { x: p.x - o.nodeWidth / 2, y: p.y - o.nodeHeight / 2 } };
  });
}

/* ------------------------------------------------------------------ */
/*  Radial — concentric circles from root nodes                       */
/* ------------------------------------------------------------------ */

function radialLayout(nodes: Node[], edges: Edge[], o: Required<LayoutOptions>): Node[] {
  if (nodes.length === 0) return [];

  const incoming = new Map<string, number>();
  nodes.forEach((n) => incoming.set(n.id, 0));
  edges.forEach((e) => incoming.set(e.target, (incoming.get(e.target) ?? 0) + 1));

  const roots = nodes.filter((n) => (incoming.get(n.id) ?? 0) === 0);
  if (roots.length === 0) roots.push(nodes[0]);

  const children = new Map<string, string[]>();
  nodes.forEach((n) => children.set(n.id, []));
  edges.forEach((e) => children.get(e.source)?.push(e.target));

  const layers = new Map<string, number>();
  const queue: string[] = [];
  roots.forEach((r) => { layers.set(r.id, 0); queue.push(r.id); });

  while (queue.length > 0) {
    const cur = queue.shift()!;
    for (const c of children.get(cur) ?? []) {
      if (!layers.has(c)) { layers.set(c, layers.get(cur)! + 1); queue.push(c); }
    }
  }
  nodes.forEach((n) => { if (!layers.has(n.id)) layers.set(n.id, 0); });

  const groups = new Map<number, string[]>();
  layers.forEach((l, id) => { if (!groups.has(l)) groups.set(l, []); groups.get(l)!.push(id); });

  const rStep = o.spacing + o.nodeWidth;

  return nodes.map((n) => {
    const layer = layers.get(n.id) ?? 0;
    const grp = groups.get(layer) ?? [n.id];
    const idx = grp.indexOf(n.id);
    if (layer === 0 && grp.length === 1) {
      return { ...n, position: { x: o.centerX - o.nodeWidth / 2, y: o.centerY - o.nodeHeight / 2 } };
    }
    const radius = layer * rStep || rStep * 0.5;
    const angle = ((2 * Math.PI) / grp.length) * idx - Math.PI / 2;
    return {
      ...n,
      position: { x: o.centerX + radius * Math.cos(angle) - o.nodeWidth / 2, y: o.centerY + radius * Math.sin(angle) - o.nodeHeight / 2 },
    };
  });
}

/* ------------------------------------------------------------------ */
/*  Mind-map — horizontal tree (Reingold-Tilford style, left→right)   */
/* ------------------------------------------------------------------ */

function mindmapLayout(nodes: Node[], edges: Edge[], o: Required<LayoutOptions>): Node[] {
  if (nodes.length === 0) return [];

  /* Build adjacency */
  const childMap = new Map<string, string[]>();
  const parentMap = new Map<string, string[]>();
  nodes.forEach((n) => { childMap.set(n.id, []); parentMap.set(n.id, []); });
  edges.forEach((e) => { childMap.get(e.source)?.push(e.target); parentMap.get(e.target)?.push(e.source); });

  /* Find roots (no incoming edges) */
  const roots = nodes.filter((n) => (parentMap.get(n.id) ?? []).length === 0);
  if (roots.length === 0) roots.push(nodes[0]);

  /* BFS to build a proper tree (handles DAG — each node visited once) */
  const treeChildren = new Map<string, string[]>();
  nodes.forEach((n) => treeChildren.set(n.id, []));
  const bfsVisited = new Set<string>();
  const bfsQueue: string[] = [];

  for (const r of roots) {
    if (!bfsVisited.has(r.id)) { bfsVisited.add(r.id); bfsQueue.push(r.id); }
  }
  while (bfsQueue.length > 0) {
    const cur = bfsQueue.shift()!;
    for (const child of childMap.get(cur) ?? []) {
      if (!bfsVisited.has(child)) {
        bfsVisited.add(child);
        treeChildren.get(cur)!.push(child);
        bfsQueue.push(child);
      }
    }
  }
  /* Add disconnected nodes as additional roots */
  for (const n of nodes) {
    if (!bfsVisited.has(n.id)) { roots.push(n); bfsVisited.add(n.id); }
  }

  /* Subtree height calculation (memoized) */
  const heightCache = new Map<string, number>();
  const vGap = Math.max(o.spacing * 0.45, 24);
  const hGap = o.spacing + o.nodeWidth * 0.15;

  function subtreeHeight(id: string): number {
    if (heightCache.has(id)) return heightCache.get(id)!;
    const children = treeChildren.get(id) ?? [];
    if (children.length === 0) { heightCache.set(id, o.nodeHeight); return o.nodeHeight; }
    const total = children.reduce((sum, c) => sum + subtreeHeight(c), 0) + (children.length - 1) * vGap;
    const h = Math.max(o.nodeHeight, total);
    heightCache.set(id, h);
    return h;
  }

  /* Position subtree recursively (children vertically centered on parent) */
  const positions = new Map<string, { x: number; y: number }>();

  function positionSubtree(id: string, x: number, yCenter: number) {
    positions.set(id, { x, y: yCenter });
    const children = treeChildren.get(id) ?? [];
    if (children.length === 0) return;

    const heights = children.map((c) => subtreeHeight(c));
    const totalHeight = heights.reduce((s, h) => s + h, 0) + (children.length - 1) * vGap;
    let yCur = yCenter - totalHeight / 2 + heights[0] / 2;
    const childX = x + o.nodeWidth + hGap;

    for (let i = 0; i < children.length; i++) {
      positionSubtree(children[i], childX, yCur);
      if (i < children.length - 1) yCur += heights[i] / 2 + vGap + heights[i + 1] / 2;
    }
  }

  /* Place each root tree, stacked vertically */
  let globalY = o.spacing + o.nodeHeight / 2;
  for (const root of roots) {
    const h = subtreeHeight(root.id);
    positionSubtree(root.id, o.spacing, globalY + (h - o.nodeHeight) / 2);
    globalY += h + o.spacing;
  }

  /* Handle any remaining unplaced nodes */
  for (const n of nodes) {
    if (!positions.has(n.id)) { positions.set(n.id, { x: o.spacing, y: globalY }); globalY += o.nodeHeight + vGap; }
  }

  return nodes.map((n) => {
    const p = positions.get(n.id)!;
    return { ...n, position: { x: p.x, y: p.y - o.nodeHeight / 2 } };
  });
}

/* ------------------------------------------------------------------ */
/*  Grid layout                                                       */
/* ------------------------------------------------------------------ */

function gridLayout(nodes: Node[], _edges: Edge[], o: Required<LayoutOptions>): Node[] {
  const cols = Math.max(1, Math.ceil(Math.sqrt(nodes.length)));
  const cw = o.nodeWidth + o.spacing;
  const ch = o.nodeHeight + o.spacing;
  return nodes.map((n, i) => ({
    ...n,
    position: { x: (i % cols) * cw + o.spacing, y: Math.floor(i / cols) * ch + o.spacing },
  }));
}

/* ------------------------------------------------------------------ */
/*  Dagre hierarchical (LR / TB)                                      */
/* ------------------------------------------------------------------ */

function hierarchicalLayout(nodes: Node[], edges: Edge[], o: Required<LayoutOptions>, dir: "LR" | "TB"): Node[] {
  if (nodes.length === 0) return [];
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({
    rankdir: dir,
    nodesep: dir === "LR" ? 40 : 50,
    ranksep: dir === "LR" ? 80 : 70,
    marginx: 40,
    marginy: 40,
    align: "UL",
  });
  nodes.forEach((n) => g.setNode(n.id, { width: o.nodeWidth, height: o.nodeHeight }));
  edges.forEach((e) => g.setEdge(e.source, e.target));
  dagre.layout(g);
  return nodes.map((n) => {
    const p = g.node(n.id);
    return { ...n, position: { x: p.x - o.nodeWidth / 2, y: p.y - o.nodeHeight / 2 } };
  });
}

/* ------------------------------------------------------------------ */
/*  Dispatcher — pick algorithm, return positioned nodes              */
/* ------------------------------------------------------------------ */

export function applyLayout(
  algorithm: LayoutAlgorithm,
  nodes: Node[],
  edges: Edge[],
  options?: LayoutOptions,
): Node[] {
  if (algorithm === "manual" || nodes.length === 0) return nodes;
  const o: Required<LayoutOptions> = { ...DEFAULTS, ...options };
  switch (algorithm) {
    case "force": return forceLayout(nodes, edges, o);
    case "radial": return radialLayout(nodes, edges, o);
    case "mindmap": return mindmapLayout(nodes, edges, o);
    case "grid": return gridLayout(nodes, edges, o);
    case "hierarchical-lr": return hierarchicalLayout(nodes, edges, o, "LR");
    case "hierarchical-tb": return hierarchicalLayout(nodes, edges, o, "TB");
    default: return nodes;
  }
}

/** Return the React Flow edge type that best suits the algorithm. */
export function getEdgeType(algorithm: LayoutAlgorithm): string {
  switch (algorithm) {
    case "force":
    case "radial":
      return "gradientEdge";
    case "mindmap":
      return "smoothstep"; // clean tree edges for horizontal tree layout
    case "grid":
      return "straight";
    default:
      return "smoothstep";
  }
}

/** Registry of available layout algorithms for the UI. */
export const LAYOUT_ALGORITHMS: { value: LayoutAlgorithm; label: string }[] = [
  { value: "mindmap", label: "Mind Map" },
  { value: "force", label: "Force" },
  { value: "radial", label: "Radial" },
  { value: "hierarchical-lr", label: "Horizontal" },
  { value: "hierarchical-tb", label: "Vertical" },
  { value: "grid", label: "Grid" },
  { value: "manual", label: "Manual" },
];
