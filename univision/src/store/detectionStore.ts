import { create } from "zustand";
import type { Detection, DetectionFilters, WsDetectionEvent } from "../types/api";
import { fetchDetections } from "../services/api";
import {
  connectDetectionEvents,
  onDetectionEvent,
  onEventsConnectionChange,
  disconnectDetectionEvents,
} from "../services/websocket";

const MAX_LIVE_EVENTS = 200;

interface DetectionState {
  /** Paginated detections from REST API */
  items: Detection[];
  total: number;
  page: number;
  pageSize: number;
  filters: DetectionFilters;
  loading: boolean;

  /** Real-time events from WebSocket */
  liveEvents: WsDetectionEvent[];
  wsConnected: boolean;

  setFilters: (filters: Partial<DetectionFilters>) => void;
  fetch: () => Promise<void>;
  nextPage: () => void;
  prevPage: () => void;

  /** Start WebSocket listener */
  startLive: () => () => void;
}

export const useDetectionStore = create<DetectionState>((set, get) => ({
  items: [],
  total: 0,
  page: 1,
  pageSize: 25,
  filters: {},
  loading: false,
  liveEvents: [],
  wsConnected: false,

  setFilters: (partial) => {
    set((s) => ({ filters: { ...s.filters, ...partial }, page: 1 }));
    get().fetch();
  },

  fetch: async () => {
    set({ loading: true });
    try {
      const { filters, page, pageSize } = get();
      const data = await fetchDetections({ ...filters, page, page_size: pageSize });
      set({ items: data.items, total: data.total, page: data.page, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  nextPage: () => {
    const { page, total, pageSize } = get();
    if (page * pageSize < total) {
      set({ page: page + 1 });
      get().fetch();
    }
  },

  prevPage: () => {
    const { page } = get();
    if (page > 1) {
      set({ page: page - 1 });
      get().fetch();
    }
  },

  startLive: () => {
    connectDetectionEvents();

    const unsub1 = onDetectionEvent((event) => {
      set((s) => ({
        liveEvents: [event, ...s.liveEvents].slice(0, MAX_LIVE_EVENTS),
      }));
    });

    const unsub2 = onEventsConnectionChange((connected) => {
      set({ wsConnected: connected });
    });

    return () => {
      unsub1();
      unsub2();
      disconnectDetectionEvents();
    };
  },
}));
