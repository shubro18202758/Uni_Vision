import { create } from "zustand";
import type { PipelineStageDef, PipelineStageEvent, PipelineStageStatus } from "../types/api";
import {
  connectPipelineStream,
  disconnectPipelineStream,
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
}

interface PipelineMonitorState {
  // Connection
  connected: boolean;

  // Current frame being processed
  currentFrame: FrameProcessingState | null;

  // Recent completed frames (ring buffer, last 20)
  recentFrames: FrameProcessingState[];

  // Stage definitions (received from backend)
  stageDefinitions: PipelineStageDef[];

  // Pipeline metrics
  queueDepth: number;
  throttled: boolean;
  totalProcessed: number;
  fps: number;

  // Flag event for auto-transition
  lastFlag: {
    frame_id: string;
    detection: Record<string, unknown>;
    validation_status: string;
    timestamp: number;
  } | null;
  flagConsumed: boolean;

  // Actions
  startMonitoring: () => () => void;
  consumeFlag: () => void;
}

const MAX_RECENT = 20;
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

export const usePipelineMonitorStore = create<PipelineMonitorState>((set, get) => ({
  connected: false,
  currentFrame: null,
  recentFrames: [],
  stageDefinitions: [],
  queueDepth: 0,
  throttled: false,
  totalProcessed: 0,
  fps: 0,
  lastFlag: null,
  flagConsumed: false,

  startMonitoring: () => {
    connectPipelineStream();

    const unsubStatus = onPipelineConnectionChange((connected) => {
      set({ connected });
    });

    const unsubEvent = onPipelineEvent((event: PipelineStageEvent) => {
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
          };

          set({
            currentFrame: newFrame,
            queueDepth: event.queue_depth ?? state.queueDepth,
            totalProcessed: (event.details?.total_frames_processed as number) ?? state.totalProcessed,
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
          set({
            currentFrame: {
              ...state.currentFrame,
              stages,
              current_stage_index: event.stage_index ?? state.currentFrame.current_stage_index,
            },
          });
          break;
        }

        case "pipeline_complete": {
          if (!state.currentFrame || state.currentFrame.frame_id !== event.frame_id) break;
          _completionTimestamps.push(Date.now());
          const completed: FrameProcessingState = {
            ...state.currentFrame,
            total_latency_ms: event.latency_ms ?? null,
            status: "complete",
          };
          const recent = [completed, ...state.recentFrames].slice(0, MAX_RECENT);
          set({
            currentFrame: null,
            recentFrames: recent,
            fps: calculateFps(),
            totalProcessed: (event.details?.total_frames_processed as number) ?? state.totalProcessed + 1,
          });
          break;
        }

        case "flag_raised": {
          if (!state.currentFrame || state.currentFrame.frame_id !== event.frame_id) break;
          const flagged: FrameProcessingState = {
            ...state.currentFrame,
            status: "flagged",
            detection: event.detection,
          };
          const recent = [flagged, ...state.recentFrames].slice(0, MAX_RECENT);
          set({
            currentFrame: flagged,
            recentFrames: recent,
            lastFlag: {
              frame_id: event.frame_id,
              detection: event.detection ?? {},
              validation_status: event.details?.validation_status as string ?? "unknown",
              timestamp: event.timestamp,
            },
            flagConsumed: false,
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
      }
    });

    return () => {
      unsubStatus();
      unsubEvent();
      disconnectPipelineStream();
      set({ connected: false, currentFrame: null });
    };
  },

  consumeFlag: () => set({ flagConsumed: true }),
}));
