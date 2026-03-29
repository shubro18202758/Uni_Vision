import { useCodeStore } from "../../store/codeStore";

export function CodeToolbar() {
  const code = useCodeStore((s) => s.code);

  const handleCopy = async () => {
    if (!code) return;
    await navigator.clipboard.writeText(code);
  };

  const handleDownload = () => {
    if (!code) return;
    const blob = new Blob([code], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "pipeline.py";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex gap-2">
      <button
        onClick={handleCopy}
        disabled={!code}
        className="rounded-md border border-slate-700/50 bg-[#111e36] px-3 py-1.5 text-xs font-semibold text-slate-300 hover:bg-[#152640] transition-all focus:ring-2 focus:ring-slate-500/10 outline-none disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Copy
      </button>
      <button
        onClick={handleDownload}
        disabled={!code}
        className="rounded-md border border-slate-700/50 bg-[#111e36] px-3 py-1.5 text-xs font-semibold text-slate-300 hover:bg-[#152640] transition-all focus:ring-2 focus:ring-slate-500/10 outline-none disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Download
      </button>
    </div>
  );
}
