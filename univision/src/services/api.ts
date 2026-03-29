/** Typed HTTP client for the Uni_Vision backend API. */

import type {
  AgentAuditEntry,
  AgentMonitorData,
  AgentStatus,
  CameraSource,
  CameraSourceInput,
  ChatRequest,
  ChatResponse,
  DeployResponse,
  DetectionFilters,
  DetectionPage,
  FeedbackRequest,
  FeedbackResponse,
  FlagReasoningResponse,
  HealthResponse,
  ImpactAnalysisResponse,
  PipelineStats,
  RiskAnalysisResponse,
  SessionSummary,
  ValidationIssue,
  VideoUploadResponse,
  VideoUploadEntry,
} from "../types/api";

import type { BlockDefinition } from "../types/block";
import type { ProjectGraph } from "../types/graph";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, body || res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ── Health ───────────────────────────────────────────────────────

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

// ── Pipeline Stats ───────────────────────────────────────────────

export function fetchStats(): Promise<PipelineStats> {
  return request<PipelineStats>("/stats");
}

// ── Detections ───────────────────────────────────────────────────

export function fetchDetections(filters: DetectionFilters = {}): Promise<DetectionPage> {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(filters)) {
    if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  }
  const qs = params.toString();
  return request<DetectionPage>(`/detections${qs ? `?${qs}` : ""}`);
}

// ── Camera Sources ───────────────────────────────────────────────

export function fetchSources(): Promise<CameraSource[]> {
  return request<CameraSource[]>("/sources");
}

export function registerSource(body: CameraSourceInput): Promise<{ camera_id: string; status: string }> {
  return request("/sources", { method: "POST", body: JSON.stringify(body) });
}

export function deleteSource(cameraId: string): Promise<void> {
  return request(`/sources/${encodeURIComponent(cameraId)}`, { method: "DELETE" });
}

// ── Video Uploads ────────────────────────────────────────────────

export async function uploadVideo(
  file: File,
  opts: { camera_id?: string; location_tag?: string; fps_target?: number } = {},
): Promise<VideoUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  if (opts.camera_id) form.append("camera_id", opts.camera_id);
  if (opts.location_tag) form.append("location_tag", opts.location_tag);
  if (opts.fps_target !== undefined) form.append("fps_target", String(opts.fps_target));

  const res = await fetch("/sources/upload", { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, body || res.statusText);
  }
  return res.json() as Promise<VideoUploadResponse>;
}

export function listUploads(): Promise<VideoUploadEntry[]> {
  return request<VideoUploadEntry[]>("/sources/upload");
}

export function deleteUpload(filename: string): Promise<{ deleted: string; status: string }> {
  return request(`/sources/upload/${encodeURIComponent(filename)}`, { method: "DELETE" });
}

// ── Pipeline Processing ──────────────────────────────────────────

import type { ProcessVideoRequest, ProcessVideoResponse, ProcessingJobStatus } from "../types/api";

export function processVideo(body: ProcessVideoRequest): Promise<ProcessVideoResponse> {
  return request<ProcessVideoResponse>("/pipeline/process", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getProcessingStatus(): Promise<ProcessingJobStatus[]> {
  return request<ProcessingJobStatus[]>("/pipeline/process/status");
}

export function stopProcessingJob(jobId: string): Promise<{ job_id: string; status: string }> {
  return request(`/pipeline/process/${encodeURIComponent(jobId)}`, { method: "DELETE" });
}

export function stopAllProcessing(): Promise<{ stopped: string[]; count: number }> {
  return request("/pipeline/process", { method: "DELETE" });
}

// ── Agent Chat ───────────────────────────────────────────────────

export function sendAgentChat(body: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/api/agent/chat", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function fetchAgentStatus(): Promise<AgentStatus> {
  return request<AgentStatus>("/api/agent/status");
}

export function submitFeedback(body: FeedbackRequest): Promise<FeedbackResponse> {
  return request<FeedbackResponse>("/api/agent/feedback", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function fetchAgentSessions(): Promise<{ sessions: SessionSummary[]; total: number }> {
  return request("/api/agent/sessions");
}

export function deleteAgentSession(sessionId: string): Promise<void> {
  return request(`/api/agent/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
}

export function fetchAgentMonitor(): Promise<AgentMonitorData> {
  return request<AgentMonitorData>("/api/agent/monitor");
}

export function fetchAgentAudit(): Promise<AgentAuditEntry[]> {
  return request<AgentAuditEntry[]>("/api/agent/audit");
}

// ── Pipeline Graph / Block Registry ──────────────────────────────

export function fetchBlocks(): Promise<BlockDefinition[]> {
  return request<BlockDefinition[]>("/api/pipeline/blocks");
}

export function fetchCategories(): Promise<Record<string, string>> {
  return request<Record<string, string>>("/api/pipeline/categories");
}

export function fetchPortTypes(): Promise<Record<string, string>> {
  return request<Record<string, string>>("/api/pipeline/port-types");
}

export function deployGraph(graph: ProjectGraph): Promise<DeployResponse> {
  return request<DeployResponse>("/api/pipeline/graph", {
    method: "POST",
    body: JSON.stringify(graph),
  });
}

export function fetchDeployedGraph(): Promise<ProjectGraph> {
  return request<ProjectGraph>("/api/pipeline/graph");
}

export function clearDeployedGraph(): Promise<{ cleared: boolean }> {
  return request("/api/pipeline/graph", { method: "DELETE" });
}

export function validateGraph(graph: ProjectGraph): Promise<ValidationIssue[]> {
  return request<ValidationIssue[]>("/api/pipeline/graph/validate", {
    method: "POST",
    body: JSON.stringify(graph),
  });
}

export function registerCustomBlock(block: BlockDefinition): Promise<{ registered: string }> {
  return request("/api/pipeline/blocks", {
    method: "POST",
    body: JSON.stringify(block),
  });
}

// ── Workflow Design ──────────────────────────────────────────────

import type { DesignWorkflowRequest, DesignWorkflowResponse } from "../types/api";

export function designWorkflow(body: DesignWorkflowRequest): Promise<DesignWorkflowResponse> {
  return request<DesignWorkflowResponse>("/api/agent/design-workflow", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// ── Model Routing (VRAM-exclusive Ollama swap) ───────────────────

export interface ModelStateResponse {
  phase: string;
  active_model: string | null;
  navarasa_loaded: boolean;
  qwen_loaded: boolean;
}

export function fetchModelState(): Promise<ModelStateResponse> {
  return request<ModelStateResponse>("/api/pipeline/model-state");
}

export function activateModel(phase: "pre_launch" | "post_launch"): Promise<ModelStateResponse> {
  return request<ModelStateResponse>("/api/pipeline/activate-model", {
    method: "POST",
    body: JSON.stringify({ phase }),
  });
}

// ── Flag Reasoning & Risk Analysis ───────────────────────────────

export function fetchFlagReasoning(detectionId: string): Promise<FlagReasoningResponse> {
  return request<FlagReasoningResponse>(
    `/api/analysis/flag-reasoning/${encodeURIComponent(detectionId)}`,
  );
}

export function fetchRiskAnalysis(detectionId: string): Promise<RiskAnalysisResponse> {
  return request<RiskAnalysisResponse>(
    `/api/analysis/risk-analysis/${encodeURIComponent(detectionId)}`,
  );
}

export function fetchImpactAnalysis(detectionId: string): Promise<ImpactAnalysisResponse> {
  return request<ImpactAnalysisResponse>(
    `/api/analysis/impact-analysis/${encodeURIComponent(detectionId)}`,
  );
}

// ── Databricks Integrations ──────────────────────────────────────

export interface DatabricksOverview {
  enabled: boolean;
  delta?: Record<string, unknown>;
  mlflow?: Record<string, unknown>;
  spark?: Record<string, unknown>;
  vector?: Record<string, unknown>;
}

export interface VectorSearchResult {
  plate_text: string;
  similarity: number;
  camera_id: string;
  confidence: number;
  engine: string;
  validation_status: string;
  timestamp: number;
}

export function fetchDatabricksOverview(): Promise<DatabricksOverview> {
  return request<DatabricksOverview>("/api/databricks/overview");
}

export function fetchDeltaStats(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/api/databricks/delta/stats");
}

export function fetchDeltaHistory(limit = 20): Promise<{ versions: Record<string, unknown>[] }> {
  return request<{ versions: Record<string, unknown>[] }>(`/api/databricks/delta/history?limit=${limit}`);
}

export function fetchMLflowSummary(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/api/databricks/mlflow/summary");
}

export function fetchMLflowMetricHistory(metric: string, limit = 50): Promise<{ metric: string; history: Record<string, unknown>[] }> {
  return request<{ metric: string; history: Record<string, unknown>[] }>(
    `/api/databricks/mlflow/metrics/${encodeURIComponent(metric)}?limit=${limit}`,
  );
}

export function fetchSparkAnalytics(queryType: string, params: Record<string, unknown> = {}): Promise<{ query_type: string; results: Record<string, unknown>[]; count: number }> {
  return request<{ query_type: string; results: Record<string, unknown>[]; count: number }>(
    "/api/databricks/spark/analytics",
    { method: "POST", body: JSON.stringify({ query_type: queryType, ...params }) },
  );
}

export function fetchSparkOverview(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/api/databricks/spark/overview");
}

export function searchSimilarPlates(query: string, topK = 20, threshold = 0.65): Promise<{ query: string; results: VectorSearchResult[]; count: number }> {
  return request<{ query: string; results: VectorSearchResult[]; count: number }>(
    "/api/databricks/vector/search",
    { method: "POST", body: JSON.stringify({ query, top_k: topK, threshold }) },
  );
}

export function fetchVectorStats(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/api/databricks/vector/stats");
}

export function fetchVectorDuplicates(threshold = 0.85): Promise<{ duplicates: Record<string, unknown>[]; count: number }> {
  return request<{ duplicates: Record<string, unknown>[]; count: number }>(
    `/api/databricks/vector/duplicates?threshold=${threshold}`,
  );
}

export function fetchVectorClusters(nClusters = 8): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(
    `/api/databricks/vector/clusters?n_clusters=${nClusters}`,
  );
}

export function searchPlatesByTimeRange(
  query: string, startTs: number, endTs: number, topK = 20, threshold = 0.65,
): Promise<{ query: string; results: VectorSearchResult[]; count: number }> {
  return request<{ query: string; results: VectorSearchResult[]; count: number }>(
    "/api/databricks/vector/search/time-range",
    { method: "POST", body: JSON.stringify({ query, start_ts: startTs, end_ts: endTs, top_k: topK, threshold }) },
  );
}

export function compactDeltaLake(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/api/databricks/delta/compact", { method: "POST" });
}

export function fetchDatabricksHealth(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/api/databricks/health");
}

export { ApiError };
