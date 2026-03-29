/** WebSocket connection managers for real-time event streams. */

import type { AgentStreamFrame, PipelineStageEvent, WsDetectionEvent } from "../types/api";

type Unsubscribe = () => void;

// ── Detection Events WebSocket (/ws/events) ──────────────────────

let eventsWs: WebSocket | null = null;
let eventsReconnectTimer: ReturnType<typeof setTimeout> | null = null;
const eventsListeners = new Set<(event: WsDetectionEvent) => void>();
const eventsStatusListeners = new Set<(connected: boolean) => void>();

function getWsBase(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}`;
}

export function connectDetectionEvents(): void {
  if (eventsWs?.readyState === WebSocket.OPEN) return;
  eventsWs?.close();

  const ws = new WebSocket(`${getWsBase()}/ws/events`);
  eventsWs = ws;

  ws.onopen = () => {
    eventsStatusListeners.forEach((fn) => fn(true));
  };

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data) as WsDetectionEvent;
      eventsListeners.forEach((fn) => fn(data));
    } catch {
      /* ignore malformed frames */
    }
  };

  ws.onclose = () => {
    eventsStatusListeners.forEach((fn) => fn(false));
    eventsReconnectTimer = setTimeout(connectDetectionEvents, 3000);
  };

  ws.onerror = () => ws.close();
}

export function onDetectionEvent(cb: (event: WsDetectionEvent) => void): Unsubscribe {
  eventsListeners.add(cb);
  return () => eventsListeners.delete(cb);
}

export function onEventsConnectionChange(cb: (connected: boolean) => void): Unsubscribe {
  eventsStatusListeners.add(cb);
  return () => eventsStatusListeners.delete(cb);
}

export function disconnectDetectionEvents(): void {
  if (eventsReconnectTimer) clearTimeout(eventsReconnectTimer);
  eventsWs?.close();
  eventsWs = null;
}

// ── Agent Reasoning WebSocket (/ws/agent) ────────────────────────

let agentWs: WebSocket | null = null;
let agentReconnectTimer: ReturnType<typeof setTimeout> | null = null;
const agentListeners = new Set<(frame: AgentStreamFrame) => void>();
const agentStatusListeners = new Set<(connected: boolean) => void>();

export function connectAgentStream(): void {
  if (agentWs?.readyState === WebSocket.OPEN) return;
  agentWs?.close();

  const ws = new WebSocket(`${getWsBase()}/ws/agent`);
  agentWs = ws;

  ws.onopen = () => {
    agentStatusListeners.forEach((fn) => fn(true));
  };

  ws.onmessage = (e) => {
    try {
      const frame = JSON.parse(e.data) as AgentStreamFrame;
      agentListeners.forEach((fn) => fn(frame));
    } catch {
      /* ignore malformed frames */
    }
  };

  ws.onclose = () => {
    agentStatusListeners.forEach((fn) => fn(false));
    agentReconnectTimer = setTimeout(connectAgentStream, 3000);
  };

  ws.onerror = () => ws.close();
}

export function sendAgentMessage(message: string, sessionId?: string): void {
  if (agentWs?.readyState !== WebSocket.OPEN) return;
  agentWs.send(JSON.stringify({ message, session_id: sessionId }));
}

export function sendWorkflowDesignMessage(
  description: string,
  language: string = "auto",
  sessionId?: string,
): void {
  if (agentWs?.readyState !== WebSocket.OPEN) return;
  agentWs.send(
    JSON.stringify({
      type: "design_workflow",
      message: description,
      language,
      session_id: sessionId,
    }),
  );
}

export function onAgentStream(cb: (frame: AgentStreamFrame) => void): Unsubscribe {
  agentListeners.add(cb);
  return () => agentListeners.delete(cb);
}

export function onAgentConnectionChange(cb: (connected: boolean) => void): Unsubscribe {
  agentStatusListeners.add(cb);
  return () => agentStatusListeners.delete(cb);
}

export function disconnectAgentStream(): void {
  if (agentReconnectTimer) clearTimeout(agentReconnectTimer);
  agentWs?.close();
  agentWs = null;
}

// ── Pipeline Visibility WebSocket (/ws/pipeline) ─────────────────

let pipelineWs: WebSocket | null = null;
let pipelineReconnectTimer: ReturnType<typeof setTimeout> | null = null;
const pipelineListeners = new Set<(event: PipelineStageEvent) => void>();
const pipelineStatusListeners = new Set<(connected: boolean) => void>();

export function connectPipelineStream(): void {
  if (pipelineWs?.readyState === WebSocket.OPEN) return;
  pipelineWs?.close();

  const ws = new WebSocket(`${getWsBase()}/ws/pipeline`);
  pipelineWs = ws;

  ws.onopen = () => {
    pipelineStatusListeners.forEach((fn) => fn(true));
  };

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data) as PipelineStageEvent;
      pipelineListeners.forEach((fn) => fn(data));
    } catch {
      /* ignore malformed frames */
    }
  };

  ws.onclose = () => {
    pipelineStatusListeners.forEach((fn) => fn(false));
    pipelineReconnectTimer = setTimeout(connectPipelineStream, 3000);
  };

  ws.onerror = () => ws.close();
}

export function onPipelineEvent(cb: (event: PipelineStageEvent) => void): Unsubscribe {
  pipelineListeners.add(cb);
  return () => pipelineListeners.delete(cb);
}

export function onPipelineConnectionChange(cb: (connected: boolean) => void): Unsubscribe {
  pipelineStatusListeners.add(cb);
  return () => pipelineStatusListeners.delete(cb);
}

export function disconnectPipelineStream(): void {
  if (pipelineReconnectTimer) clearTimeout(pipelineReconnectTimer);
  pipelineWs?.close();
  pipelineWs = null;
}
