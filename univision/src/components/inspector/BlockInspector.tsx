import { useState } from "react";
import { getBlockDefinition } from "../../lib/blockRegistry";
import { useGraphStore } from "../../store/graphStore";
import { ConfigFieldRenderer } from "./ConfigFieldRenderer";
import { getCategoryColor } from "../../constants/categories";
import { ChevronDown, ChevronRight, MessageSquareText, Lightbulb, ArrowRight, ArrowLeft } from "lucide-react";

/** Computer-vision pipeline instruction presets per category. */
const INSTRUCTION_PRESETS: Record<string, string[]> = {
  Input: [
    "Connect to the parking lot entrance RTSP camera and stream at 5 FPS",
    "Load test images from the captured plate samples folder",
    "Stream from highway toll booth camera with H.265 encoding",
  ],
  Ingestion: [
    "Sample every 3rd frame to balance coverage and processing load",
    "Drop frames when GPU utilisation exceeds 85%",
    "Buffer 10 frames and release the sharpest one per batch",
  ],
  Detection: [
    "Detect all vehicles — cars, trucks, buses, motorcycles — with ≥50% confidence",
    "Detect anomalies or objects of interest with tight bounding boxes",
    "Use YOLOv8 nano model for real-time detection on 1080p streams",
  ],
  Preprocessing: [
    "Deskew and enhance contrast on plate crops for better OCR accuracy",
    "Apply CLAHE histogram equalisation and resize plates to 200px width",
    "Sharpen blurry night-time plate captures before OCR",
  ],
  OCR: [
    "Read plate numbers with high accuracy using angle classifier",
    "Use PaddleOCR with high-accuracy mode for text extraction",
    "Extract text from multi-line plates with confidence scoring",
  ],
  PostProcessing: [
    "Validate plates against Indian format XX00XX0000",
    "Remove duplicate readings within a 10-second window using perceptual hash",
    "Normalise OCR output — strip spaces, fix common O/0 confusions",
  ],
  Output: [
    "Save validated plates to database and broadcast via Redis pub/sub",
    "Draw bounding boxes and plate labels on the live monitoring feed",
    "Export detection logs as CSV every hour",
  ],
  Utility: [
    "Log pipeline throughput metrics to Prometheus",
    "Add a 500ms delay between processing batches",
    "Route frames to different branches based on confidence score",
  ],
};

export function BlockInspector() {
  const selectedBlockId = useGraphStore((state) => state.selectedBlockId);
  const blocks = useGraphStore((state) => state.blocks);
  const updateBlockConfig = useGraphStore((state) => state.updateBlockConfig);
  const updateBlockLabel = useGraphStore((state) => state.updateBlockLabel);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [presetsOpen, setPresetsOpen] = useState(false);

  const block = blocks.find((item) => item.id === selectedBlockId);
  const definition = block ? getBlockDefinition(block.type) : undefined;

  if (!block || !definition) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 p-8 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-800/60 border border-slate-700/40">
          <ArrowLeft size={18} className="text-slate-500" />
        </div>
        <p className="text-sm text-slate-500">Select a block on the canvas to configure it.</p>
        <p className="text-[10px] text-slate-600">Drag blocks from the palette or double-click the canvas</p>
      </div>
    );
  }

  const catColor = getCategoryColor(block.category);
  const instructionField = definition.configSchema.find((f) => f.key === "instruction");
  const advancedFields = definition.configSchema.filter((f) => f.key !== "instruction");
  const presets = INSTRUCTION_PRESETS[block.category] ?? [];

  return (
    <div className="space-y-5 p-5">
      {/* Category badge */}
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: catColor }} />
        <p className="text-[10px] font-bold uppercase tracking-[0.3em] text-slate-500">{block.category}</p>
      </div>

      {/* Editable label */}
      <div>
        <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-1 block">Block Name</label>
        <input
          className="w-full text-lg font-bold tracking-tight text-slate-100 bg-transparent border-b-2 border-transparent hover:border-slate-700 focus:border-slate-500 outline-none pb-1 transition-colors"
          value={block.label}
          onChange={(e) => updateBlockLabel(block.id, e.target.value)}
        />
      </div>

      {/* Description */}
      <p className="text-xs leading-relaxed text-slate-500">{definition.description}</p>

      {/* Port summary */}
      <div className="flex items-center gap-3">
        {definition.inputs.length > 0 && (
          <div className="flex items-center gap-1.5 text-[10px] text-blue-400/80">
            <ArrowRight size={10} />
            <span className="font-bold">{definition.inputs.length}</span>
            <span className="text-slate-500">input{definition.inputs.length > 1 ? "s" : ""}</span>
            <span className="text-slate-600">({definition.inputs.map((p) => p.name).join(", ")})</span>
          </div>
        )}
        {definition.outputs.length > 0 && (
          <div className="flex items-center gap-1.5 text-[10px] text-amber-400/80">
            <ArrowLeft size={10} />
            <span className="font-bold">{definition.outputs.length}</span>
            <span className="text-slate-500">output{definition.outputs.length > 1 ? "s" : ""}</span>
          </div>
        )}
      </div>

      {/* ── Instruction — Hero field ── */}
      {instructionField && (
        <div className="rounded-xl border border-slate-700/40 bg-slate-800/20 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <MessageSquareText size={16} className="text-slate-400" />
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Your Instruction</h3>
            </div>
            {presets.length > 0 && (
              <button
                onClick={() => setPresetsOpen(!presetsOpen)}
                className="flex items-center gap-1 rounded-md px-2 py-1 text-[9px] font-bold text-amber-400/80 hover:bg-amber-950/30 transition-colors"
              >
                <Lightbulb size={10} />
                Suggestions
                {presetsOpen ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
              </button>
            )}
          </div>

          {/* Instruction presets */}
          {presetsOpen && presets.length > 0 && (
            <div className="mb-3 space-y-1">
              {presets.map((preset) => (
                <button
                  key={preset}
                  onClick={() => {
                    updateBlockConfig(block.id, "instruction", preset);
                    setPresetsOpen(false);
                  }}
                  className="w-full text-left rounded-lg border border-amber-800/20 bg-amber-950/20 px-3 py-2 text-[10px] text-amber-300/80 hover:border-amber-600/40 hover:bg-amber-950/40 transition-all"
                >
                  {preset}
                </button>
              ))}
            </div>
          )}

          <textarea
            className="w-full rounded-lg bg-[#0a1628] border border-slate-700/40 px-3 py-3 text-sm text-slate-200 leading-relaxed placeholder:text-slate-600 focus:ring-2 focus:ring-slate-500/30 focus:border-slate-500/50 outline-none resize-y min-h-[80px] transition-all"
            rows={5}
            placeholder={instructionField.placeholder ?? "Describe what this step should do in plain English..."}
            value={String(block.config.instruction ?? "")}
            onChange={(e) => updateBlockConfig(block.id, "instruction", e.target.value)}
          />
          <p className="text-[10px] text-slate-600 mt-2 leading-relaxed">
            Write what you want this step to do. The AI pipeline will interpret your intent when launched.
          </p>
        </div>
      )}

      {/* ── Advanced Settings — collapsible ── */}
      {advancedFields.length > 0 && (
        <div className="rounded-xl border border-slate-700/50 overflow-hidden">
          <button
            className="w-full flex items-center justify-between px-4 py-3 bg-[#0a1628] hover:bg-slate-800/60 transition-colors text-left"
            onClick={() => setAdvancedOpen(!advancedOpen)}
          >
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              Advanced Settings
            </span>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-slate-600 font-medium">{advancedFields.length} fields</span>
              {advancedOpen ? (
                <ChevronDown size={14} className="text-slate-500" />
              ) : (
                <ChevronRight size={14} className="text-slate-500" />
              )}
            </div>
          </button>
          {advancedOpen && (
            <div className="space-y-4 p-4 border-t border-slate-700/50">
              {advancedFields.map((field) => (
                <ConfigFieldRenderer
                  key={field.key}
                  field={field}
                  onChange={(value) => updateBlockConfig(block.id, field.key, value)}
                  value={block.config[field.key]}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
