import { create } from "zustand";
import type { CameraSource, CameraSourceInput, VideoUploadEntry, VideoUploadResponse } from "../types/api";
import { fetchSources, registerSource, deleteSource, uploadVideo, listUploads, deleteUpload } from "../services/api";

interface CameraState {
  sources: CameraSource[];
  uploads: VideoUploadEntry[];
  loading: boolean;
  uploading: boolean;
  uploadProgress: string | null;
  error: string | null;

  fetch: () => Promise<void>;
  add: (input: CameraSourceInput) => Promise<void>;
  remove: (cameraId: string) => Promise<void>;
  uploadFile: (file: File, opts?: { camera_id?: string; location_tag?: string; fps_target?: number }) => Promise<VideoUploadResponse | null>;
  fetchUploads: () => Promise<void>;
  removeUpload: (filename: string) => Promise<void>;
}

export const useCameraStore = create<CameraState>((set, get) => ({
  sources: [],
  uploads: [],
  loading: false,
  uploading: false,
  uploadProgress: null,
  error: null,

  fetch: async () => {
    set({ loading: true, error: null });
    try {
      const sources = await fetchSources();
      set({ sources, loading: false });
    } catch (e) {
      set({ loading: false, error: e instanceof Error ? e.message : "Failed to load cameras" });
    }
  },

  add: async (input) => {
    set({ error: null });
    try {
      await registerSource(input);
      await get().fetch();
    } catch (e) {
      set({ error: e instanceof Error ? e.message : "Failed to register camera" });
    }
  },

  remove: async (cameraId) => {
    set({ error: null });
    try {
      await deleteSource(cameraId);
      set((s) => ({ sources: s.sources.filter((c) => c.camera_id !== cameraId) }));
    } catch (e) {
      set({ error: e instanceof Error ? e.message : "Failed to delete camera" });
    }
  },

  uploadFile: async (file, opts = {}) => {
    set({ uploading: true, uploadProgress: `Uploading ${file.name}...`, error: null });
    try {
      const result = await uploadVideo(file, opts);
      set({ uploading: false, uploadProgress: null });
      // Refresh both lists
      await get().fetch();
      await get().fetchUploads();
      return result;
    } catch (e) {
      set({
        uploading: false,
        uploadProgress: null,
        error: e instanceof Error ? e.message : "Upload failed",
      });
      return null;
    }
  },

  fetchUploads: async () => {
    try {
      const uploads = await listUploads();
      set({ uploads });
    } catch {
      // Non-critical — don't overwrite main error
    }
  },

  removeUpload: async (filename) => {
    set({ error: null });
    try {
      await deleteUpload(filename);
      set((s) => ({ uploads: s.uploads.filter((u) => u.filename !== filename) }));
    } catch (e) {
      set({ error: e instanceof Error ? e.message : "Failed to delete upload" });
    }
  },
}));
