import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, Plus, Trash2, RefreshCw, Wifi, WifiOff, Video, Upload, Film, X, FileVideo } from "lucide-react";
import { useCameraStore } from "../../store/cameraStore";
import type { CameraSourceInput } from "../../types/api";
import clsx from "clsx";

const EMPTY: CameraSourceInput = { camera_id: "", source_url: "", location_tag: "", fps_target: 3, enabled: true };

const ACCEPTED_FORMATS = ".mp4,.avi,.mkv,.webm,.mov,.flv,.mpeg,.mpg,.3gp,.wmv,.ogv,.ts";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function CameraManager() {
  const sources = useCameraStore((s) => s.sources);
  const uploads = useCameraStore((s) => s.uploads);
  const loading = useCameraStore((s) => s.loading);
  const uploading = useCameraStore((s) => s.uploading);
  const uploadProgress = useCameraStore((s) => s.uploadProgress);
  const error = useCameraStore((s) => s.error);
  const fetch_ = useCameraStore((s) => s.fetch);
  const add = useCameraStore((s) => s.add);
  const remove = useCameraStore((s) => s.remove);
  const uploadFile = useCameraStore((s) => s.uploadFile);
  const fetchUploads = useCameraStore((s) => s.fetchUploads);
  const removeUpload = useCameraStore((s) => s.removeUpload);

  const [form, setForm] = useState<CameraSourceInput>(EMPTY);
  const [showForm, setShowForm] = useState(false);
  const [tab, setTab] = useState<"cameras" | "upload">("cameras");
  const [dragOver, setDragOver] = useState(false);
  const [uploadOpts, setUploadOpts] = useState({ camera_id: "", location_tag: "", fps_target: 3 });
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { fetch_(); fetchUploads(); }, [fetch_, fetchUploads]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.camera_id || !form.source_url) return;
    await add(form);
    setForm(EMPTY);
    setShowForm(false);
  };

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      await uploadFile(file, {
        camera_id: uploadOpts.camera_id || undefined,
        location_tag: uploadOpts.location_tag || undefined,
        fps_target: uploadOpts.fps_target,
      });
    }
    // Reset camera_id so next upload gets a fresh auto-ID
    setUploadOpts((o) => ({ ...o, camera_id: "" }));
  }, [uploadFile, uploadOpts]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => setDragOver(false), []);

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Camera size={14} className="text-slate-400" />
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
            Camera Sources
          </span>
        </div>
        <div className="flex gap-1">
          <button onClick={fetch_} className="rounded p-1 hover:bg-slate-800" title="Refresh">
            <RefreshCw size={12} className={clsx("text-slate-400", loading && "animate-spin")} />
          </button>
          <button onClick={() => setShowForm(!showForm)} className="rounded p-1 hover:bg-slate-800" title="Add camera">
            <Plus size={12} className="text-slate-400" />
          </button>
        </div>
      </div>

      {/* Tabs: Cameras | Upload Video */}
      <div className="flex border-b border-slate-700/40">
        <button
          onClick={() => setTab("cameras")}
          className={clsx(
            "flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider transition-colors",
            tab === "cameras" ? "border-b-2 border-cyan-500 text-cyan-400" : "text-slate-500 hover:text-slate-300",
          )}
        >
          <span className="flex items-center justify-center gap-1"><Wifi size={10} /> Live</span>
        </button>
        <button
          onClick={() => setTab("upload")}
          className={clsx(
            "flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider transition-colors",
            tab === "upload" ? "border-b-2 border-violet-500 text-violet-400" : "text-slate-500 hover:text-slate-300",
          )}
        >
          <span className="flex items-center justify-center gap-1"><Upload size={10} /> Upload</span>
        </button>
      </div>

      {error && (
        <div className="rounded border border-rose-800/40 bg-rose-950/30 px-2 py-1 text-[10px] text-rose-400">{error}</div>
      )}

      {/* ── Cameras Tab ── */}
      {tab === "cameras" && (
        <>
          {/* Add form */}
          {showForm && (
            <form onSubmit={handleSubmit} className="space-y-2 rounded-md border border-slate-700/30 bg-slate-800/20 p-3">
              <input
                value={form.camera_id}
                onChange={(e) => setForm({ ...form, camera_id: e.target.value })}
                placeholder="Camera ID (e.g., cam-front-01)"
                className="w-full rounded border border-slate-700/50 bg-[#0a1628] px-2 py-1 text-[11px] text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-slate-500"
              />
              <input
                value={form.source_url}
                onChange={(e) => setForm({ ...form, source_url: e.target.value })}
                placeholder="RTSP URL or file path"
                className="w-full rounded border border-slate-700/50 bg-[#0a1628] px-2 py-1 text-[11px] text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-slate-500"
              />
              <div className="flex gap-2">
                <input
                  value={form.location_tag}
                  onChange={(e) => setForm({ ...form, location_tag: e.target.value })}
                  placeholder="Location tag"
                  className="flex-1 rounded border border-slate-700/50 bg-[#0a1628] px-2 py-1 text-[11px] text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-slate-500"
                />
                <input
                  type="number"
                  value={form.fps_target}
                  onChange={(e) => setForm({ ...form, fps_target: Number(e.target.value) })}
                  placeholder="FPS"
                  min={1} max={30}
                  className="w-16 rounded border border-slate-700/50 bg-[#0a1628] px-2 py-1 text-[11px] text-slate-200 focus:outline-none focus:border-slate-500"
                />
              </div>
              <div className="flex gap-2">
                <button type="submit" className="rounded bg-slate-600 px-3 py-1 text-[10px] font-bold text-white hover:bg-slate-500">
                  Register
                </button>
                <button type="button" onClick={() => setShowForm(false)} className="text-[10px] text-slate-500 hover:text-slate-300">
                  Cancel
                </button>
              </div>
            </form>
          )}

          {/* Camera list */}
          <div className="flex-1 overflow-auto space-y-1.5">
            {loading && sources.length === 0 && (
              <p className="text-center text-xs text-slate-400 py-8">Loading cameras...</p>
            )}
            {!loading && sources.length === 0 && (
              <div className="flex flex-col items-center gap-2 py-8 text-slate-400">
                <Video size={24} />
                <p className="text-xs">No cameras registered</p>
                <button onClick={() => setShowForm(true)} className="text-[10px] text-slate-400 hover:underline">
                  Add your first camera
                </button>
              </div>
            )}
            {sources.map((src) => (
              <div key={src.camera_id} className="flex items-center justify-between rounded-md border border-slate-700/40 bg-[#111e36] px-3 py-2 hover:bg-[#152640] transition-colors">
                <div className="flex flex-col min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    {src.enabled ? (
                      <Wifi size={10} className="text-emerald-500 flex-shrink-0" />
                    ) : (
                      <WifiOff size={10} className="text-slate-500 flex-shrink-0" />
                    )}
                    <span className="text-xs font-bold text-slate-200 truncate">{src.camera_id}</span>
                  </div>
                  <span className="text-[9px] text-slate-400 truncate">{src.source_url}</span>
                  {src.location_tag && (
                    <span className="text-[9px] text-slate-500">{src.location_tag} &middot; {src.fps_target} fps</span>
                  )}
                </div>
                <button
                  onClick={() => remove(src.camera_id)}
                  className="flex-shrink-0 rounded p-1 text-slate-500 hover:bg-rose-950/50 hover:text-rose-400 transition-colors"
                  title="Remove camera"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        </>
      )}

      {/* ── Upload Tab ── */}
      {tab === "upload" && (
        <div className="flex flex-1 flex-col gap-3 overflow-auto">
          {/* Drop zone */}
          <div
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => fileInputRef.current?.click()}
            className={clsx(
              "flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed p-6 transition-colors",
              dragOver
                ? "border-violet-400 bg-violet-950/30"
                : "border-slate-700/50 bg-[#0a1628] hover:border-slate-500 hover:bg-slate-800/30",
              uploading && "pointer-events-none opacity-60",
            )}
          >
            {uploading ? (
              <>
                <div className="h-6 w-6 animate-spin rounded-full border-2 border-violet-500 border-t-transparent" />
                <p className="text-[11px] text-violet-400">{uploadProgress || "Uploading..."}</p>
              </>
            ) : (
              <>
                <Upload size={24} className="text-slate-500" />
                <p className="text-[11px] text-slate-400">
                  <span className="font-semibold text-violet-400">Click to browse</span> or drag &amp; drop
                </p>
                <p className="text-[9px] text-slate-600">
                  MP4, AVI, MKV, WebM, MOV, FLV, MPEG, 3GP, WMV, OGV, TS — up to 500 MB
                </p>
              </>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_FORMATS}
              multiple
              className="hidden"
              title="Select video files to upload"
              onChange={(e) => handleFiles(e.target.files)}
            />
          </div>

          {/* Upload options */}
          <div className="space-y-2 rounded-md border border-slate-700/30 bg-slate-800/20 p-3">
            <p className="text-[9px] font-bold uppercase tracking-wider text-slate-500">Upload Options</p>
            <input
              value={uploadOpts.camera_id}
              onChange={(e) => setUploadOpts({ ...uploadOpts, camera_id: e.target.value })}
              placeholder="Camera ID (auto-generated if empty)"
              className="w-full rounded border border-slate-700/50 bg-[#0a1628] px-2 py-1 text-[11px] text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-slate-500"
            />
            <div className="flex gap-2">
              <input
                value={uploadOpts.location_tag}
                onChange={(e) => setUploadOpts({ ...uploadOpts, location_tag: e.target.value })}
                placeholder="Location tag"
                className="flex-1 rounded border border-slate-700/50 bg-[#0a1628] px-2 py-1 text-[11px] text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-slate-500"
              />
              <input
                type="number"
                value={uploadOpts.fps_target}
                onChange={(e) => setUploadOpts({ ...uploadOpts, fps_target: Number(e.target.value) })}
                min={1} max={30}
                placeholder="FPS"
                className="w-16 rounded border border-slate-700/50 bg-[#0a1628] px-2 py-1 text-[11px] text-slate-200 focus:outline-none focus:border-slate-500"
                title="FPS target"
              />
            </div>
          </div>

          {/* Uploaded videos list */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <p className="text-[9px] font-bold uppercase tracking-wider text-slate-500">Uploaded Videos</p>
              <button onClick={fetchUploads} className="rounded p-1 hover:bg-slate-800" title="Refresh uploads">
                <RefreshCw size={10} className="text-slate-500" />
              </button>
            </div>
            {uploads.length === 0 && (
              <div className="flex flex-col items-center gap-2 py-4 text-slate-500">
                <FileVideo size={20} />
                <p className="text-[10px]">No videos uploaded yet</p>
              </div>
            )}
            {uploads.map((u) => (
              <div key={u.filename} className="flex items-center justify-between rounded-md border border-slate-700/40 bg-[#111e36] px-3 py-2 hover:bg-[#152640] transition-colors">
                <div className="flex flex-col min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <Film size={10} className="text-violet-400 flex-shrink-0" />
                    <span className="text-xs font-bold text-slate-200 truncate" title={u.filename}>
                      {u.filename.length > 30 ? u.filename.slice(13) : u.filename}
                    </span>
                  </div>
                  <span className="text-[9px] text-slate-500">
                    {u.format.toUpperCase()} &middot; {formatBytes(u.size_bytes)}
                  </span>
                </div>
                <button
                  onClick={() => removeUpload(u.filename)}
                  className="flex-shrink-0 rounded p-1 text-slate-500 hover:bg-rose-950/50 hover:text-rose-400 transition-colors"
                  title="Delete video"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
