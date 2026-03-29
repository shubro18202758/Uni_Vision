interface Props {
  label: string;
  value: string | number | boolean | undefined;
  onChange: (value: number) => void;
}

export function NumberField({ label, value, onChange }: Props) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs uppercase tracking-[0.2em] text-slate-400">{label}</span>
      <input
        className="w-full rounded-md border border-slate-700/50 bg-[#0a1628] px-3 py-2 text-sm text-slate-200 outline-none focus:ring-2 focus:ring-slate-500/20 focus:border-slate-500 transition-all"
        onChange={(event) => onChange(Number(event.target.value))}
        type="number"
        value={String(value ?? 0)}
      />
    </label>
  );
}
