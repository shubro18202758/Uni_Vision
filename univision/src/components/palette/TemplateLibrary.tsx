import { STARTER_TEMPLATES } from "../../constants/templates";
import { useGraphStore } from "../../store/graphStore";

export function TemplateLibrary() {
  const setGraph = useGraphStore((state) => state.setGraph);

  return (
    <section className="rounded-lg border border-slate-700/40 bg-[#0d1b2e] p-4 shadow-lg shadow-black/10">
      <h2 className="text-[10px] font-bold uppercase tracking-[0.25em] text-slate-500">Starter Blueprints</h2>
      <div className="mt-3 space-y-2">
        {STARTER_TEMPLATES.map((template) => (
          <button
            key={template.project.name}
            className="w-full rounded-lg border border-slate-700/50 bg-[#111e36] p-4 text-left transition hover:border-slate-500/50 hover:bg-[#152640] hover:shadow-md hover:shadow-black/10 group"
            onClick={() => setGraph(template)}
            type="button"
          >
            <p className="text-[11px] font-bold text-slate-300 group-hover:text-slate-100 transition-colors">{template.project.name}</p>
            <p className="mt-1 text-[10px] text-slate-500">Load pipeline blueprint</p>
          </button>
        ))}
      </div>
    </section>
  );
}
