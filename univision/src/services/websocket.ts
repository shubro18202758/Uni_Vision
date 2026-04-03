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
  if (eventsWs?.readyState === WebSocket.OPEN || eventsWs?.readyState === WebSocket.CONNECTING) return;
  eventsWs?.close();

  const ws = new WebSocket(`${getWsBase()}/ws/events`);
  eventsWs = ws;

  ws.onopen = () => {
    eventsStatusListeners.forEach((fn) => { try { fn(true); } catch { /* */ } });
  };

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data) as WsDetectionEvent;
      eventsListeners.forEach((fn) => { try { fn(data); } catch { /* */ } });
    } catch {
      /* ignore malformed frames */
    }
  };

  ws.onclose = () => {
    eventsStatusListeners.forEach((fn) => { try { fn(false); } catch { /* */ } });
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
  if (eventsReconnectTimer) { clearTimeout(eventsReconnectTimer); eventsReconnectTimer = null; }
  if (eventsWs) {
    eventsWs.onclose = null;
    eventsWs.onerror = null;
    eventsWs.onmessage = null;
    eventsWs.close();
    eventsWs = null;
  }
}

// ── Agent Reasoning WebSocket (/ws/agent) ────────────────────────

let agentWs: WebSocket | null = null;
let agentReconnectTimer: ReturnType<typeof setTimeout> | null = null;
const agentListeners = new Set<(frame: AgentStreamFrame) => void>();
const agentStatusListeners = new Set<(connected: boolean) => void>();

export function connectAgentStream(): void {
  if (agentWs?.readyState === WebSocket.OPEN || agentWs?.readyState === WebSocket.CONNECTING) return;
  agentWs?.close();

  const ws = new WebSocket(`${getWsBase()}/ws/agent`);
  agentWs = ws;

  ws.onopen = () => {
    agentStatusListeners.forEach((fn) => { try { fn(true); } catch { /* */ } });
  };

  ws.onmessage = (e) => {
    try {
      const frame = JSON.parse(e.data) as AgentStreamFrame;
      agentListeners.forEach((fn) => { try { fn(frame); } catch { /* */ } });
    } catch {
      /* ignore malformed frames */
    }
  };

  ws.onclose = () => {
    agentStatusListeners.forEach((fn) => { try { fn(false); } catch { /* */ } });
    agentReconnectTimer = setTimeout(connectAgentStream, 3000);
  };

  ws.onerror = () => ws.close();
}

export function sendAgentMessage(message: string, sessionId?: string): void {
  if (agentWs?.readyState !== WebSocket.OPEN) return;
  agentWs.send(JSON.stringify({ message, session_id: sessionId }));
}

function waitForAgentWsOpen(timeoutMs = 5000): Promise<boolean> {
  return new Promise((resolve) => {
    if (agentWs?.readyState === WebSocket.OPEN) { resolve(true); return; }
    if (!agentWs || agentWs.readyState === WebSocket.CLOSED || agentWs.readyState === WebSocket.CLOSING) {
      resolve(false); return;
    }
    const timer = setTimeout(() => { resolve(false); }, timeoutMs);
    const onOpen = () => { clearTimeout(timer); agentWs?.removeEventListener("open", onOpen); resolve(true); };
    agentWs.addEventListener("open", onOpen);
  });
}

export async function sendWorkflowDesignMessage(
  description: string,
  language: string = "auto",
  sessionId?: string,
): Promise<void> {
  const ready = await waitForAgentWsOpen();
  if (!ready || agentWs?.readyState !== WebSocket.OPEN) return;
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
  if (agentReconnectTimer) { clearTimeout(agentReconnectTimer); agentReconnectTimer = null; }
  if (agentWs) {
    agentWs.onclose = null;
    agentWs.onerror = null;
    agentWs.onmessage = null;
    agentWs.close();
    agentWs = null;
  }
}

// ── Pipeline Visibility WebSocket (/ws/pipeline) ─────────────────

let pipelineWs: WebSocket | null = null;
let pipelineReconnectTimer: ReturnType<typeof setTimeout> | null = null;
let pipelinePingInterval: ReturnType<typeof setInterval> | null = null;
const pipelineListeners = new Set<(event: PipelineStageEvent) => void>();
const pipelineStatusListeners = new Set<(connected: boolean) => void>();

export function connectPipelineStream(): void {
  // Prevent reconnect loop: skip if already OPEN or still CONNECTING
  if (
    pipelineWs?.readyState === WebSocket.OPEN ||
    pipelineWs?.readyState === WebSocket.CONNECTING
  )
    return;
  pipelineWs?.close();

  const ws = new WebSocket(`${getWsBase()}/ws/pipeline`);
  pipelineWs = ws;

  ws.onopen = () => {
    pipelineStatusListeners.forEach((fn) => {
      try { fn(true); } catch { /* */ }
    });
    // Periodic ping every 15s to keep connection alive & detect stale links
    if (pipelinePingInterval) clearInterval(pipelinePingInterval);
    pipelinePingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        try { ws.send("ping"); } catch { /* */ }
      }
    }, 15_000);
  };

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data) as PipelineStageEvent;
      // Skip heartbeat/pong frames
      if ((data as { type?: string }).type === "heartbeat") return;
      // Error-isolated listener dispatch — one failing listener cannot stop others
      pipelineListeners.forEach((fn) => {
        try { fn(data); } catch (err) { console.error("[pipeline-ws] listener error:", err); }
      });
    } catch {
      /* ignore malformed frames (pong responses, etc.) */
    }
  };

  ws.onclose = () => {
    if (pipelinePingInterval) { clearInterval(pipelinePingInterval); pipelinePingInterval = null; }
    pipelineStatusListeners.forEach((fn) => {
      try { fn(false); } catch { /* */ }
    });
    // Auto-reconnect after 3s
    if (pipelineReconnectTimer) clearTimeout(pipelineReconnectTimer);
    pipelineReconnectTimer = setTimeout(connectPipelineStream, 3000);
  };

  ws.onerror = () => ws.close();
}

export function isPipelineConnected(): boolean {
  return pipelineWs?.readyState === WebSocket.OPEN;
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
  if (pipelineReconnectTimer) { clearTimeout(pipelineReconnectTimer); pipelineReconnectTimer = null; }
  if (pipelinePingInterval) { clearInterval(pipelinePingInterval); pipelinePingInterval = null; }
  if (pipelineWs) {
    // Remove handlers BEFORE close to prevent zombie reconnect from onclose
    pipelineWs.onclose = null;
    pipelineWs.onerror = null;
    pipelineWs.onmessage = null;
    pipelineWs.close();
    pipelineWs = null;
  }
}
