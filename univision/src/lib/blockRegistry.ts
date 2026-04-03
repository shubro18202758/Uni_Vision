/**
 * Dynamic block registry — fetches available blocks from the backend
 * API at ``/api/pipeline/blocks``.  Falls back to static defaults
 * if the backend is unreachable.
 *
 * Static defaults are loaded **synchronously** at import time so that
 * ``getBlockDefinition()`` always returns a value before React's first
 * render.  The async ``initBlockRegistry()`` call then replaces them
 * with fresh backend data when available.
 *
 * Exports a React-friendly ``useBlockRegistry()`` hook powered by
 * ``useSyncExternalStore`` so components re-render when the async
 * load completes (including after Vite HMR module replacement).
 */

import { useSyncExternalStore } from "react";
import type { BlockDefinition } from "../types/block";
import { mergeCategoryColors } from "../constants/categories";
import { mergePortColors } from "../constants/portTypes";
import { STATIC_BLOCK_DEFAULTS } from "./blockRegistryDefaults";

// ── Mutable registry state ──────────────────────────────────────
// Seeded synchronously with static defaults so blocks are never empty.

const _blocks: BlockDefinition[] = [...STATIC_BLOCK_DEFAULTS];
let _loaded = false;
let _loading: Promise<void> | null = null;

// ── Reactive subscription for React (useSyncExternalStore) ──────

let _version = 0;
const _listeners = new Set<() => void>();

function _subscribe(listener: () => void) {
  _listeners.add(listener);
  return () => _listeners.delete(listener);
}

function _getSnapshot() {
  return _version;
}

function _notify() {
  _version++;
  _listeners.forEach((l) => l());
}

/**
 * React hook — returns the current block list and re-renders the
 * component whenever the registry is (re-)loaded.
 */
export function useBlockRegistry(): BlockDefinition[] {
  useSyncExternalStore(_subscribe, _getSnapshot);
  return _blocks;
}

// ── Public API ──────────────────────────────────────────────────

/** Read-only view of the current block list (non-reactive). */
export function getBlockRegistry(): BlockDefinition[] {
  return _blocks;
}

/**
 * Kept for backward compatibility — modules that imported ``blockRegistry``
 * as a const array now get the live reference.  The array is mutated
 * in-place by ``initBlockRegistry()``.
 */
export const blockRegistry: BlockDefinition[] = _blocks;

export function getBlockDefinition(type: string): BlockDefinition | undefined {
  return _blocks.find((b) => b.type === type);
}

export function isBlockRegistryLoaded(): boolean {
  return _loaded;
}

/**
 * Fetch live block definitions from the backend.  Safe to call
 * multiple times — only the first invocation triggers the fetch.
 * If the backend is unreachable, the synchronously-loaded static
 * defaults remain in place.
 */
export async function initBlockRegistry(): Promise<void> {
  if (_loaded) return;
  if (_loading) return _loading;

  _loading = _doLoad();
  return _loading;
}

async function _doLoad(): Promise<void> {
  try {
    const [blocksRes, catsRes, portsRes] = await Promise.all([
      fetch("/api/pipeline/blocks"),
      fetch("/api/pipeline/categories"),
      fetch("/api/pipeline/port-types"),
    ]);

    if (blocksRes.ok) {
      const data: BlockDefinition[] = await blocksRes.json();
      // Ensure every block has an instruction field in configSchema.
      // The instruction textarea is the hero UI element — it must always be present.
      for (const block of data) {
        const hasInstruction = block.configSchema.some((f) => f.key === "instruction");
        if (!hasInstruction) {
          block.configSchema.unshift({
            key: "instruction",
            label: "Your Instruction",
            type: "textarea",
            placeholder: "Describe what this step should do in plain English...",
          });
        }
        if (block.defaults.instruction === undefined) {
          block.defaults.instruction = "";
        }
      }
      // Merge: static defaults as base, backend data overrides by type.
      // This ensures block types only defined in the frontend (used by the
      // workflow designer) are always available even if the backend hasn't
      // registered them yet.
      const byType = new Map<string, BlockDefinition>();
      for (const b of STATIC_BLOCK_DEFAULTS) byType.set(b.type, b);
      for (const b of data) byType.set(b.type, b);
      const merged = Array.from(byType.values());
      _blocks.length = 0;
      _blocks.push(...merged);
      blockRegistry.length = 0;
      blockRegistry.push(...merged);
    }

    if (catsRes.ok) {
      const cats: Record<string, string> = await catsRes.json();
      mergeCategoryColors(cats);
    }

    if (portsRes.ok) {
      const ports: Record<string, string> = await portsRes.json();
      mergePortColors(ports);
    }
  } catch {
    // Network error — static defaults already loaded, nothing to do.
  }

  _loaded = true;
  _loading = null;
  _notify(); // trigger React re-renders in subscribed components
}

// ── Self-init on module load ────────────────────────────────────
// Covers both initial page load and Vite HMR module replacement.
// The _loaded guard inside initBlockRegistry() prevents duplicate fetches.
void initBlockRegistry();
