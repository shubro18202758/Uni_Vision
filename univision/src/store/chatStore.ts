import { create } from "zustand";
import type { AgentStreamFrame } from "../types/api";
import { sendAgentChat } from "../services/api";
import {
  connectAgentStream,
  sendAgentMessage,
  onAgentStream,
  onAgentConnectionChange,
  disconnectAgentStream,
} from "../services/websocket";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  intent?: string;
  agentRole?: string;
  steps?: AgentStreamFrame[];
  streaming?: boolean;
}

interface ChatState {
  messages: ChatMessage[];
  sessionId: string | null;
  wsConnected: boolean;
  sending: boolean;
  streamMode: boolean;

  /** Send via REST (batch response) */
  sendChat: (message: string) => Promise<void>;

  /** Send via WebSocket (streaming) */
  sendStream: (message: string) => void;

  /** Toggle between REST and WS streaming */
  setStreamMode: (mode: boolean) => void;

  clearMessages: () => void;

  /** Start WebSocket agent listener */
  startAgentWs: () => () => void;
}

let streamBuffer: AgentStreamFrame[] = [];

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  sessionId: null,
  wsConnected: false,
  sending: false,
  streamMode: true,

  sendChat: async (message) => {
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: message,
      timestamp: Date.now(),
    };
    set((s) => ({ messages: [...s.messages, userMsg], sending: true }));

    try {
      const res = await sendAgentChat({
        message,
        session_id: get().sessionId ?? undefined,
      });
      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: res.answer,
        timestamp: Date.now(),
        intent: res.intent ?? undefined,
        agentRole: res.agent_role ?? undefined,
      };
      set((s) => ({
        messages: [...s.messages, assistantMsg],
        sessionId: res.session_id ?? s.sessionId,
        sending: false,
      }));
    } catch (e) {
      const errMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "system",
        content: e instanceof Error ? e.message : "Failed to reach agent",
        timestamp: Date.now(),
      };
      set((s) => ({ messages: [...s.messages, errMsg], sending: false }));
    }
  },

  sendStream: (message) => {
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: message,
      timestamp: Date.now(),
    };
    streamBuffer = [];
    const placeholderId = crypto.randomUUID();
    const placeholder: ChatMessage = {
      id: placeholderId,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      streaming: true,
      steps: [],
    };
    set((s) => ({ messages: [...s.messages, userMsg, placeholder], sending: true }));
    sendAgentMessage(message, get().sessionId ?? undefined);
  },

  setStreamMode: (mode) => set({ streamMode: mode }),

  clearMessages: () => set({ messages: [], sessionId: null }),

  startAgentWs: () => {
    connectAgentStream();

    const unsub1 = onAgentStream((frame) => {
      streamBuffer.push(frame);

      if (frame.type === "done" || frame.type === "error") {
        // Finalize the streaming message
        set((s) => {
          const msgs = [...s.messages];
          const idx = msgs.findIndex((m) => m.streaming);
          if (idx !== -1) {
            msgs[idx] = {
              ...msgs[idx],
              content: frame.answer || msgs[idx].content || frame.content || "",
              streaming: false,
              steps: [...streamBuffer],
              agentRole: frame.role,
            };
          } else if (frame.type === "error") {
            // Error arrived after done already finalised — append as system message
            msgs.push({
              id: crypto.randomUUID(),
              role: "system",
              content: frame.content || "Agent encountered an error",
              timestamp: Date.now(),
            });
          }
          streamBuffer = [];
          return {
            messages: msgs,
            sending: false,
            sessionId: frame.session_id ?? s.sessionId,
          };
        });
        return;
      }

      // Update streaming message in-place
      set((s) => {
        const msgs = [...s.messages];
        const idx = msgs.findIndex((m) => m.streaming);
        if (idx !== -1) {
          const contentParts: string[] = [];
          if (frame.type === "thought" && frame.content) contentParts.push(frame.content);
          if (frame.type === "answer" && frame.content) contentParts.push(frame.content);

          if (contentParts.length > 0) {
            const prev = msgs[idx].content;
            msgs[idx] = {
              ...msgs[idx],
              content: prev ? `${prev}\n${contentParts.join("\n")}` : contentParts.join("\n"),
              steps: [...streamBuffer],
            };
          }
        }
        return { messages: msgs };
      });
    });

    const unsub2 = onAgentConnectionChange((connected) => {
      set({ wsConnected: connected });
    });

    return () => {
      unsub1();
      unsub2();
      disconnectAgentStream();
    };
  },
}));
