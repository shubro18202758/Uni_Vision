/** Backend API response types — mirrors FastAPI Pydantic schemas */

// ── Health ───────────────────────────────────────────────────────

export interface HealthResponse {
  healthy: boolean;
  mode?: string;
  gpu_available?: boolean;
  ollama_reachable?: boolean;
  database_connected?: boolean;
  streams_connected?: number;
  streams_total?: number;
  offload_mode?: string;
  details?: Record<string, string>;
}

// ── Detections ───────────────────────────────────────────────────

export interface Detection {
  id: string;
  camera_id: string;
  plate_number: string;
  raw_ocr_text: string;
  ocr_confidence: number;
  ocr_engine: string;
  vehicle_class: string;
  vehicle_image_path: string;
  plate_image_path: string;
  detected_at_utc: string | null;
  validation_status: string;
  location_tag: string;
}

export interface DetectionPage {
  items: Detection[];
  total: number;
  page: number;
  page_size: number;
}

export interface DetectionFilters {
  camera_id?: string;
  plate_number?: string;
  status?: string;
  since?: string;
  until?: string;
  page?: number;
  page_size?: number;
}

// ── Camera Sources ───────────────────────────────────────────────

export interface CameraSource {
  camera_id: string;
  source_url: string;
  location_tag: string;
  fps_target: number;
  enabled: boolean;
  added_at?: string | null;
}

export interface CameraSourceInput {
  camera_id: string;
  source_url: string;
  location_tag?: string;
  fps_target?: number;
  enabled?: boolean;
}

// ── Video Uploads ────────────────────────────────────────────────

export interface VideoUploadResponse {
  camera_id: string;
  filename: string;
  original_name: string;
  size_bytes: number;
  source_url: string;
  format: string;
  fps_target: number;
  location_tag: string;
  db_registered: boolean;
  status: string;
}

export interface VideoUploadEntry {
  filename: string;
  size_bytes: number;
  format: string;
  source_url: string;
}

// ── Pipeline Processing ──────────────────────────────────────────

export interface ProcessVideoRequest {
  source_url: string;
  camera_id?: string;
  location_tag?: string;
  fps_target?: number;
}

export interface ProcessVideoResponse {
  job_id: string;
  camera_id: string;
  source_url: string;
  status: string;
}

export interface ProcessingJobStatus {
  job_id: string;
  camera_id: string;
  source_url: string;
  status: string;
}

// ── Agent Chat ───────────────────────────────────────────────────

export interface ChatRequest {
  message: string;
  session_id?: string;
}

export interface StepDetail {
  step: number;
  thought: string;
  tool?: string | null;
  observation?: string | null;
  answer?: string | null;
  elapsed_ms: number;
}

export interface ChatResponse {
  answer: string;
  steps: StepDetail[];
  total_steps: number;
  elapsed_ms: number;
  success: boolean;
  error?: string | null;
  session_id?: string | null;
  intent?: string | null;
  agent_role?: string | null;
}

export interface AgentStatus {
  running: boolean;
  tool_count: number;
  available_tools: string[];
}

export interface FeedbackRequest {
  detection_id: string;
  feedback_type: "confirm" | "correct" | "reject";
  original_text: string;
  corrected_text?: string;
  camera_id?: string;
  notes?: string;
}

// ── WebSocket Events ─────────────────────────────────────────────

export interface AnomalyItem {
  type: string;
  description: string;
  severity: string;
  location: string;
}

export interface DetectedObject {
  label: string;
  location: string;
  condition: string;
}

export interface WsDetectionEvent {
  id: string;
  camera_id: string;
  detected_at_utc: string;
  validation_status: string;
  confidence: number;
  // Generic analysis fields
  scene_description?: string;
  objects_detected?: DetectedObject[];
  anomaly_detected?: boolean;
  anomalies?: AnomalyItem[];
  chain_of_thought?: string;
  risk_level?: string;
  risk_analysis?: string;
  impact_analysis?: string;
  recommendations?: string[];
  // Legacy detection fields (optional, backward compat)
  plate_number?: string;
  raw_ocr_text?: string;
  ocr_confidence?: number;
  ocr_engine?: string;
  vehicle_class?: string;
  vehicle_image_path?: string;
  plate_image_path?: string;
  location_tag?: string;
}

export type AgentStreamType =
  | "intent"
  | "thought"
  | "tool_call"
  | "observation"
  | "answer"
  | "done"
  | "error"
  | "workflow_lock"
  | "workflow_phase"
  | "workflow_complete";

export interface AgentStreamFrame {
  type: AgentStreamType;
  step?: number;
  content?: string;
  tool?: string;
  args?: Record<string, unknown>;
  intent?: string;
  role?: string;
  confidence?: number;
  answer?: string;
  total_steps?: number;
  elapsed_ms?: number;
  success?: boolean;
  session_id?: string;
  /** Workflow design streaming fields */
  locked?: boolean;
  phase?: string;
  message?: string;
  graph?: Record<string, unknown>;
  phases?: WorkflowPhaseDetail[];
  detected_language?: string;
  english_input?: string;
  original_input?: string;
  error?: string;
  total_elapsed_ms?: number;
}

// ── Workflow Design ──────────────────────────────────────────────

export interface WorkflowPhaseDetail {
  name: string;
  message: string;
  elapsed_ms: number;
  success: boolean;
}

export interface DesignWorkflowRequest {
  description: string;
  language?: string;
  session_id?: string;
}

export interface DesignWorkflowResponse {
  success: boolean;
  graph?: Record<string, unknown>;
  phases: WorkflowPhaseDetail[];
  detected_language?: string;
  english_input?: string;
  original_input?: string;
  error?: string;
  total_elapsed_ms: number;
}

// ── Pipeline Stats ───────────────────────────────────────────────

export type PipelineStats = Record<string, number>;

// ── Sessions ─────────────────────────────────────────────────────

export interface SessionSummary {
  session_id: string;
  created_at: string;
  last_active: string;
  turn_count: number;
  idle_seconds: number;
}

// ── Feedback Response ────────────────────────────────────────────

export interface FeedbackResponse {
  status: string;
  feedback_type: string;
  detection_id: string;
}

// ── Agent Monitor / Audit ────────────────────────────────────────

export interface AgentMonitorData {
  sessions_active: number;
  total_turns: number;
  avg_latency_ms: number;
  error_rate: number;
}

export interface AgentAuditEntry {
  id: string;
  session_id: string;
  action: string;
  timestamp: string;
  details: Record<string, unknown>;
}

// ── Pipeline Graph / Block Registry ──────────────────────────────

export interface DeployResponse {
  success: boolean;
  issues: ValidationIssue[];
  deployed_nodes: number;
  deployed_edges: number;
}

export interface ValidationIssue {
  id: string;
  level: "error" | "warning";
  message: string;
  blockId?: string;
}

// ── Flag Reasoning & Risk Analysis ───────────────────────────────

export interface EvidenceItem {
  evidence_type: string;
  label: string;
  description: string;
  metric_value: number | null;
  threshold: number | null;
  severity: string;
  raw_data: Record<string, unknown>;
}

export interface FlagReasoningResponse {
  detection_id: string;
  flagged: boolean;
  message?: string;
  flag_type?: string;
  severity?: string;
  headline?: string;
  reasoning_chain?: string[];
  evidence?: EvidenceItem[];
  alert_count?: number;
  confidence_score?: number;
  generated_at_ms?: number;
}

export interface RiskDimension {
  axis: string;
  score: number;
  label: string;
  description: string;
  trend: string;
}

export interface TimelineEvent {
  timestamp: string;
  event_type: string;
  title: string;
  description: string;
  severity: string;
  metric_value: number | null;
}

export interface ScenarioProjection {
  scenario: string;
  title: string;
  description: string;
  probability: number;
  impact_score: number;
  time_to_resolution: string;
  consequences_if_ignored: string[];
  escalation_severity: string;
}

export interface AnomalyPattern {
  pattern_name: string;
  frequency: number;
  affected_cameras: string[];
  time_distribution: Record<string, number>;
  severity_distribution: Record<string, number>;
}

export interface ComponentHealth {
  component: string;
  health_pct: number;
  latency_score: number;
  reliability_score: number;
  accuracy_score: number;
  status: string;
}

export interface AlertItem {
  alert_id: string;
  priority: string;
  title: string;
  description: string;
  source_component: string;
  metric_value: number | null;
  threshold: number | null;
}

export interface ConsequenceStep {
  step: number;
  timeframe: string;
  event: string;
  description: string;
  severity: string;
  probability: number;
  affected_components: string[];
}

export interface IgnoredAlertConsequence {
  alert_id: string;
  alert_title: string;
  consequence_chain: ConsequenceStep[];
  terminal_state: string;
  total_propagation_time: string;
  cascading_failure_risk: number;
}

export interface RiskAnalysisResponse {
  detection_id: string;
  overall_risk_level: string;
  overall_risk_score: number;
  risk_dimensions: RiskDimension[];
  timeline: TimelineEvent[];
  scenarios: ScenarioProjection[];
  anomaly_patterns: AnomalyPattern[];
  component_health: ComponentHealth[];
  alerts: AlertItem[];
  ignored_consequences: IgnoredAlertConsequence[];
  summary: string;
  generated_at_ms: number;
}

// ── Impact Analysis ──────────────────────────────────────────────

export interface ImpactDimension {
  domain: string;
  title: string;
  score: number;
  severity: string;
  description: string;
  affected_components: string[];
  metrics: Record<string, unknown>;
}

export interface TemporalImpactPoint {
  time_offset: string;
  time_seconds: number;
  cumulative_frames_lost: number;
  surveillance_coverage_pct: number;
  data_quality_pct: number;
  system_stability_pct: number;
  detection_accuracy_pct: number;
  operator_trust_pct: number;
  description: string;
}

export interface CascadeNode {
  component: string;
  failure_mode: string;
  time_to_failure: string;
  probability: number;
  downstream: string[];
  severity: string;
}

export interface ResourceImpact {
  resource: string;
  current_usage_pct: number;
  projected_peak_pct: number;
  headroom_pct: number;
  time_to_exhaustion: string;
  risk_level: string;
}

export interface CoverageGap {
  camera_id: string;
  gap_type: string;
  start_offset: string;
  duration_estimate: string;
  vehicles_missed_estimate: number;
  zone_affected: string;
  severity: string;
}

export interface DataCorruptionVector {
  source: string;
  destination: string;
  data_type: string;
  corruption_type: string;
  records_affected_estimate: number;
  forensic_impact: string;
  severity: string;
}

export interface ComplianceImpact {
  regulation: string;
  requirement: string;
  current_status: string;
  time_to_violation: string;
  liability_level: string;
  description: string;
}

export interface FunnelStage {
  stage: string;
  total_frames: number;
  successful: number;
  failed: number;
  drop_rate_pct: number;
  bottleneck: boolean;
}

export interface HeatmapCell {
  component: string;
  time_bucket: string;
  health_pct: number;
  anomaly_count: number;
}

export interface CorrelationPair {
  metric_a: string;
  metric_b: string;
  correlation: number;
  relationship: string;
}

export interface ImpactAnalysisResponse {
  detection_id: string;
  overall_impact_score: number;
  overall_severity: string;
  impact_dimensions: ImpactDimension[];
  temporal_propagation: TemporalImpactPoint[];
  cascade_chain: CascadeNode[];
  resource_impacts: ResourceImpact[];
  coverage_gaps: CoverageGap[];
  data_corruption_vectors: DataCorruptionVector[];
  compliance_impacts: ComplianceImpact[];
  processing_funnel: FunnelStage[];
  component_heatmap: HeatmapCell[];
  correlations: CorrelationPair[];
  summary: string;
  analysis_time_ms: number;
}

// ── Pipeline Visibility / Processing Monitor ─────────────────────

export type PipelineStageStatus = "pending" | "running" | "completed" | "skipped" | "failed";

export interface PipelineStageDef {
  index: number;
  id: string;
  name: string;
  description: string;
}

export type PipelineEventType =
  | "frame_accepted"
  | "stage_started"
  | "stage_completed"
  | "frame_preview"
  | "pipeline_complete"
  | "flag_raised"
  | "pipeline_idle"
  | "queue_status"
  | "analysis_result"
  | "job_created"
  | "job_phase_changed"
  | "component_provisioned"
  | "job_flushing"
  | "job_flush_complete";

export interface PipelineStageEvent {
  type: PipelineEventType;
  frame_id: string;
  camera_id: string;
  timestamp: number;
  stage_id?: string;
  stage_index?: number;
  stage_name?: string;
  stage_description?: string;
  status?: PipelineStageStatus;
  latency_ms?: number;
  total_stages: number;
  details?: Record<string, unknown>;
  thumbnail_b64?: string;
  stages_definition?: PipelineStageDef[];
  queue_depth?: number;
  detection?: Record<string, unknown>;
  analysis?: Record<string, unknown>;
  // Manager Agent lifecycle fields (job_* / component_* events)
  data?: Record<string, unknown>;
}

// ── Manager Agent Job Lifecycle ──────────────────────────────────

export type ManagerJobPhase =
  | "initializing"
  | "discovering"
  | "provisioning"
  | "processing"
  | "anomaly_detected"
  | "completing"
  | "flushing"
  | "completed"
  | "error";

export interface ManagerJobState {
  jobId: string;
  cameraId: string;
  phase: ManagerJobPhase;
  dynamicComponents: string[];
  dynamicPipPackages: string[];
  anomalyFrames: number;
  totalFrames: number;
  flushed: boolean;
  flushSummary?: Record<string, unknown>;
}

// ── Technical Metrics ────────────────────────────────────────────

export interface TechnicalBottleneck {
  component: string;
  type: string;
  description: string;
  mitigation: string;
}

export interface TechnicalMetricsResponse {
  detection_id: string;
  anomaly_type: string;
  anomaly_severity: string;
  inference_metrics: Record<string, unknown>;
  accuracy_metrics: Record<string, unknown>;
  pipeline_metrics: Record<string, unknown>;
  hardware_metrics: Record<string, unknown>;
  libraries: Record<string, string>;
  media_constraints: Record<string, unknown>;
  bottlenecks: TechnicalBottleneck[];
  generated_at_utc: string;
}
