interface Props {
  label: string;
  value: string | number | boolean | undefined;
  onChange: (value: boolean) => void;
}

export function ToggleField({ label, value, onChange }: Props) {
  return (
    <label className="flex items-center justify-between rounded-md border border-slate-700/50 bg-[#0a1628] px-4 py-3 hover:bg-[#111e36] transition-all group cursor-pointer font-medium">
      <span className="text-sm text-slate-300 group-hover:text-slate-100 transition-colors">{label}</span>
      <input 
        checked={Boolean(value)} 
        onChange={(event) => onChange(event.target.checked)} 
        type="checkbox" 
        className="h-4 w-4 rounded border-slate-700 text-slate-500 focus:ring-slate-500/20"
      />
    </label>
  );
}
