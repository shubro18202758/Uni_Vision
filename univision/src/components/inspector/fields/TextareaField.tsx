interface Props {
  label: string;
  value: string | number | boolean | undefined;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
}

export function TextareaField({ label, value, onChange, placeholder, rows = 4 }: Props) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs uppercase tracking-[0.2em] text-slate-400">{label}</span>
      <textarea
        className="w-full rounded-lg border border-slate-700/50 bg-[#0a1628] px-3 py-2.5 text-sm text-slate-200 outline-none focus:ring-2 focus:ring-slate-500/20 focus:border-slate-500 transition-all resize-y leading-relaxed placeholder:text-slate-600"
        onChange={(event) => onChange(event.target.value)}
        value={String(value ?? "")}
        placeholder={placeholder}
        rows={rows}
      />
    </label>
  );
}
