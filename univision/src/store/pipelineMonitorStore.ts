import { create } from "zustand";
import type { PipelineStageDef, PipelineStageEvent, PipelineStageStatus, ManagerJobState, ManagerJobPhase } from "../types/api";
import {
  connectPipelineStream,
  disconnectPipelineStream,
  isPipelineConnected,
  onPipelineEvent,
  onPipelineConnectionChange,
} from "../services/websocket";

// ── Stage tracking per frame ─────────────────────────────────────

export interface StageState {
  id: string;
  index: number;
  name: string;
  description: string;
  status: PipelineStageStatus;
  latency_ms: number | null;
  details: Record<string, unknown>;
  thumbnail_b64?: string;
}

export interface FrameProcessingState {
  frame_id: string;
  camera_id: string;
  started_at: number;
  stages: StageState[];
  current_stage_index: number;
  total_latency_ms: number | null;
  status: "processing" | "complete" | "flagged";
  frame_thumbnail?: string;
  detection?: Record<string, unknown>;
  /** Per-anomaly detections accumulated from individual flag_raised events */
  detections: Record<string, unknown>[];
}

// ── Real-time metrics ────────────────────────────────────────────

export interface StageLatencyPoint {
  frame_id: string;
  timestamp: number;
  latency_ms: number;
}

export interface PipelineMetrics {
  /** Per-stage latency history (last N frames per stage) */
  stageLatencyHistory: Record<string, StageLatencyPoint[]>;
  /** Total pipeline latency per frame */
  totalLatencyHistory: Array<{ frame_id: string; timestamp: number; latency_ms: number }>;
  /** FPS samples over time */
  fpsHistory: Array<{ timestamp: number; fps: number }>;
  /** Queue depth over time */
  queueHistory: Array<{ timestamp: number; depth: number }>;
  /** Avg latency per stage (rolling) */
  stageAvgLatency: Record<string, number>;
  /** Total avg pipeline latency */
  avgPipelineLatency: number;
  /** Max latency seen per stage */
  stageMaxLatency: Record<string, number>;
  /** Min latency seen per stage */
  stageMinLatency: Record<string, number>;
  /** Frames with anomalies count */
  anomalyCount: number;
  /** Processing start time */
  startedAt: number | null;
  /** Current throughput (frames/sec) */
  throughput: number;
}

interface PipelineMonitorState {
  // Connection
  connected: boolean;

  // Current frame being processed
  currentFrame: FrameProcessingState | null;

  // Last completed frame — kept visible during transitions to prevent flicker
  lastCompletedFrame: FrameProcessingState | null;

  // Recent completed frames (ring buffer, last 20)
  recentFrames: FrameProcessingState[];

  // Stage definitions (received from backend)
  stageDefinitions: PipelineStageDef[];

  // Pipeline metrics
  queueDepth: number;
  throttled: boolean;
  totalProcessed: number;
  fps: number;

  // Real-time metrics
  metrics: PipelineMetrics;

  // Flag event for auto-transition
  lastFlag: {
    frame_id: string;
    detection: Record<string, unknown>;
    validation_status: string;
    timestamp: number;
  } | null;
  flagConsumed: boolean;

  // Manager Agent job lifecycle tracking
  managerJob: ManagerJobState | null;

  // Actions
  startMonitoring: () => () => void;
  consumeFlag: () => void;
}

const MAX_RECENT = 20;
const MAX_METRIC_POINTS = 60; // ~60 data points per metric chart
const FPS_WINDOW_MS = 5000;
const _completionTimestamps: number[] = [];

function calculateFps(): number {
  const now = Date.now();
  const cutoff = now - FPS_WINDOW_MS;
  while (_completionTimestamps.length > 0 && _completionTimestamps[0] < cutoff) {
    _completionTimestamps.shift();
  }
  return Math.round((_completionTimestamps.length / (FPS_WINDOW_MS / 1000)) * 10) / 10;
}

function buildInitialStages(defs: PipelineStageDef[]): StageState[] {
  return defs.map((d) => ({
    id: d.id,
    index: d.index,
    name: d.name,
    description: d.description,
    status: "pending" as PipelineStageStatus,
    latency_ms: null,
    details: {},
  }));
}

function createEmptyMetrics(): PipelineMetrics {
  return {
    stageLatencyHistory: {},
    totalLatencyHistory: [],
    fpsHistory: [],
    queueHistory: [],
    stageAvgLatency: {},
    avgPipelineLatency: 0,
    stageMaxLatency: {},
    stageMinLatency: {},
    anomalyCount: 0,
    startedAt: null,
    throughput: 0,
  };
}

function appendCapped<T>(arr: T[], item: T, max: number = MAX_METRIC_POINTS): T[] {
  const next = [...arr, item];
  return next.length > max ? next.slice(next.length - max) : next;
}

function recalcStageAvg(history: StageLatencyPoint[]): number {
  if (history.length === 0) return 0;
  return history.reduce((s, p) => s + p.latency_ms, 0) / history.length;
}

export const usePipelineMonitorStore = create<PipelineMonitorState>((set, get) => ({
  connected: false,
  currentFrame: null,
  lastCompletedFrame: null,
  recentFrames: [],
  stageDefinitions: [],
  queueDepth: 0,
  throttled: false,
  totalProcessed: 0,
  fps: 0,
  metrics: createEmptyMetrics(),
  lastFlag: null,
  flagConsumed: false,
  managerJob: null,

  startMonitoring: () => {
    connectPipelineStream();
    // Set initial connected state from current WS (prevents stale false after zombie reconnect)
    set({ connected: isPipelineConnected() });

    const unsubStatus = onPipelineConnectionChange((connected) => {
      set({ connected });
    });

    const unsubEvent = onPipelineEvent((event: PipelineStageEvent) => {
      try {
      const state = get();

      switch (event.type) {
        case "frame_accepted": {
          // Store stage definitions if provided
          const defs = event.stages_definition ?? state.stageDefinitions;
          if (event.stages_definition && event.stages_definition.length > 0) {
            set({ stageDefinitions: event.stages_definition });
          }

          const stages = buildInitialStages(defs);
          // Mark S0 as completed (frame ingested)
          if (stages.length > 0) {
            stages[0].status = "completed";
            stages[0].thumbnail_b64 = event.thumbnail_b64;
          }

          const newFrame: FrameProcessingState = {
            frame_id: event.frame_id,
            camera_id: event.camera_id,
            started_at: event.timestamp,
            stages,
            current_stage_index: 0,
            total_latency_ms: null,
            status: "processing",
            frame_thumbnail: event.thumbnail_b64,
            detections: [],
          };

          // If previous frame was still processing, push it to recentFrames
          const prevFrame = state.currentFrame;
          let recentFrames = state.recentFrames;
          if (prevFrame && prevFrame.status === "processing") {
            const pushed: FrameProcessingState = { ...prevFrame, status: "complete" };
            recentFrames = [pushed, ...recentFrames].slice(0, MAX_RECENT);
          }

          // Track queue depth metric
          const metrics = { ...state.metrics };
          const now = Date.now();
          if (!metrics.startedAt) metrics.startedAt = now;
          metrics.queueHistory = appendCapped(metrics.queueHistory, {
            timestamp: now,
            depth: event.queue_depth ?? state.queueDepth,
          });

          set({
            currentFrame: newFrame,
            recentFrames,
            queueDepth: event.queue_depth ?? state.queueDepth,
            totalProcessed: (event.details?.total_frames_processed as number) ?? state.totalProcessed,
            metrics,
          });
          break;
        }

        case "stage_started": {
          if (!state.currentFrame || state.currentFrame.frame_id !== event.frame_id) break;
          const stages = [...state.currentFrame.stages];
          const idx = stages.findIndex((s) => s.id === event.stage_id);
          if (idx >= 0) {
            stages[idx] = { ...stages[idx], status: "running" };
          }
          set({
            currentFrame: {
              ...state.currentFrame,
              stages,
              current_stage_index: event.stage_index ?? state.currentFrame.current_stage_index,
            },
          });
          break;
        }

        case "stage_completed": {
          if (!state.currentFrame || state.currentFrame.frame_id !== event.frame_id) break;
          const stages = [...state.currentFrame.stages];
          const idx = stages.findIndex((s) => s.id === event.stage_id);
          if (idx >= 0) {
            stages[idx] = {
              ...stages[idx],
              status: (event.status as PipelineStageStatus) ?? "completed",
              latency_ms: event.latency_ms ?? null,
              details: (event.details as Record<string, unknown>) ?? {},
              thumbnail_b64: event.thumbnail_b64 ?? stages[idx].thumbnail_b64,
            };
          }

          // Track per-stage latency in metrics
          const metrics = { ...state.metrics };
          if (event.stage_id && event.latency_ms != null) {
            const sid = event.stage_id;
            const point: StageLatencyPoint = {
              frame_id: event.frame_id,
              timestamp: Date.now(),
              latency_ms: event.latency_ms,
            };
            const hist = [...(metrics.stageLatencyHistory[sid] ?? []), point];
            metrics.stageLatencyHistory = {
              ...metrics.stageLatencyHistory,
              [sid]: hist.length > MAX_METRIC_POINTS ? hist.slice(hist.length - MAX_METRIC_POINTS) : hist,
            };
            metrics.stageAvgLatency = {
              ...metrics.stageAvgLatency,
              [sid]: recalcStageAvg(metrics.stageLatencyHistory[sid]),
            };
            metrics.stageMaxLatency = {
              ...metrics.stageMaxLatency,
              [sid]: Math.max(metrics.stageMaxLatency[sid] ?? 0, event.latency_ms),
            };
            metrics.stageMinLatency = {
              ...metrics.stageMinLatency,
              [sid]: Math.min(metrics.stageMinLatency[sid] ?? Infinity, event.latency_ms),
            };
          }

          set({
            currentFrame: {
              ...state.currentFrame,
              stages,
              current_stage_index: event.stage_index ?? state.currentFrame.current_stage_index,
            },
            metrics,
          });
          break;
        }

        case "pipeline_complete": {
          if (!state.currentFrame || state.currentFrame.frame_id !== event.frame_id) break;
          _completionTimestamps.push(Date.now());
          const newFps = calculateFps();
          const completed: FrameProcessingState = {
            ...state.currentFrame,
            total_latency_ms: event.latency_ms ?? null,
            // Preserve "flagged" status if flag_raised already fired for this frame
            status: state.currentFrame.status === "flagged" ? "flagged" : "complete",
          };
          const recent = [completed, ...state.recentFrames].slice(0, MAX_RECENT);

          // Track total latency & FPS in metrics
          const metrics = { ...state.metrics };
          const now = Date.now();
          if (event.latency_ms != null) {
            metrics.totalLatencyHistory = appendCapped(metrics.totalLatencyHistory, {
              frame_id: event.frame_id,
              timestamp: now,
              latency_ms: event.latency_ms,
            });
            const totalHist = metrics.totalLatencyHistory;
            metrics.avgPipelineLatency = totalHist.reduce((s, p) => s + p.latency_ms, 0) / totalHist.length;
          }
          metrics.fpsHistory = appendCapped(metrics.fpsHistory, { timestamp: now, fps: newFps });
          if (state.metrics.startedAt) {
            const elapsed = (now - state.metrics.startedAt) / 1000;
            metrics.throughput = elapsed > 0 ? (state.totalProcessed + 1) / elapsed : 0;
          }

          set({
            currentFrame: null,
            lastCompletedFrame: completed,
            recentFrames: recent,
            fps: newFps,
            totalProcessed: (event.details?.total_frames_processed as number) ?? state.totalProcessed + 1,
            metrics,
          });
          break;
        }

        case "flag_raised": {
          if (!state.currentFrame || state.currentFrame.frame_id !== event.frame_id) break;
          const det = event.detection ?? {};
          const prevDetections = state.currentFrame.detections ?? [];
          const flagged: FrameProcessingState = {
            ...state.currentFrame,
            status: "flagged",
            detection: det,
            detections: [...prevDetections, det],
          };
          // Don't push to recentFrames here — pipeline_complete will handle it.
          // Pushing here AND in pipeline_complete caused duplicate entries.
          const metrics = { ...state.metrics, anomalyCount: state.metrics.anomalyCount + 1 };
          set({
            currentFrame: flagged,
            lastFlag: {
              frame_id: event.frame_id,
              detection: det,
              validation_status: event.details?.validation_status as string ?? "unknown",
              timestamp: event.timestamp,
            },
            flagConsumed: false,
            metrics,
          });
          break;
        }

        case "analysis_result": {
          if (!state.currentFrame || state.currentFrame.frame_id !== event.frame_id) break;
          set({
            currentFrame: {
              ...state.currentFrame,
              detection: event.analysis ?? event.details as Record<string, unknown>,
            },
          });
          break;
        }

        case "queue_status": {
          set({
            queueDepth: event.queue_depth ?? 0,
            throttled: (event.details?.throttled as boolean) ?? false,
            totalProcessed: (event.details?.total_frames_processed as number) ?? state.totalProcessed,
          });
          break;
        }

        // ── Manager Agent job lifecycle events ─────────────────

        case "job_created": {
          const d = event.data ?? event.details ?? {};
          set({
            managerJob: {
              jobId: (d.job_id as string) ?? "",
              cameraId: (d.camera_id as string) ?? "",
              phase: "initializing" as ManagerJobPhase,
              dynamicComponents: [],
              dynamicPipPackages: [],
              anomalyFrames: 0,
              totalFrames: 0,
              flushed: false,
            },
          });
          break;
        }

        case "job_phase_changed": {
          const d = event.data ?? event.details ?? {};
          const mj = state.managerJob;
          if (mj) {
            set({
              managerJob: {
                ...mj,
                phase: (d.new_phase as ManagerJobPhase) ?? mj.phase,
              },
            });
          }
          break;
        }

        case "component_provisioned": {
          const d = event.data ?? event.details ?? {};
          const mj = state.managerJob;
          if (mj) {
            const cid = (d.component_id as string) ?? "";
            const pkg = (d.pip_package as string) ?? "";
            set({
              managerJob: {
                ...mj,
                dynamicComponents: cid && !mj.dynamicComponents.includes(cid)
                  ? [...mj.dynamicComponents, cid]
                  : mj.dynamicComponents,
                dynamicPipPackages: pkg && !mj.dynamicPipPackages.includes(pkg)
                  ? [...mj.dynamicPipPackages, pkg]
                  : mj.dynamicPipPackages,
              },
            });
          }
          break;
        }

        case "job_flushing": {
          const mj = state.managerJob;
          if (mj) {
            set({ managerJob: { ...mj, phase: "flushing" } });
          }
          break;
        }

        case "job_flush_complete": {
          const d = event.data ?? event.details ?? {};
          const mj = state.managerJob;
          if (mj) {
            set({
              managerJob: {
                ...mj,
                phase: "completed",
                flushed: true,
                flushSummary: d as Record<string, unknown>,
              },
            });
          }
          break;
        }
      }
      } catch (err) {
        console.error("[pipeline-monitor] event handler error:", err);
      }
    });

    return () => {
      unsubStatus();
      unsubEvent();
      disconnectPipelineStream();
      set({ connected: false, currentFrame: null, lastCompletedFrame: null, metrics: createEmptyMetrics() });
    };
  },

  consumeFlag: () => set({ flagConsumed: true }),
}));
