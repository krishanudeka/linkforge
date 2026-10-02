import React from 'react'

export default function PaperList({ papers, onEntity }) {
  if (!papers?.length) return <p className="text-slate-500 text-sm p-3">No papers yet. Ingest some from the “Ingest” tab.</p>
  const click = (e) => {
    const ent = e.target?.dataset?.entity
    if (ent) onEntity?.(ent)
  }
  return (
    <ol className="space-y-3 p-3">
      {papers.map((p, i) => (
        <li key={p.paper_id} className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
          <div className="text-sm font-medium text-slate-100">
            {i + 1}. {p.title || p.paper_id} <span className="text-slate-500 font-normal">{p.year || ''}</span>
          </div>
          {p.path && p.path.length > 1 && (
            <div className="mt-1 text-xs text-sky-300 flex flex-wrap items-center gap-1">
              Path:
              {p.path.map((n, k) => (
                <React.Fragment key={k}>
                  {k > 0 && <span className="text-slate-500">→</span>}
                  <button className="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700" onClick={() => onEntity?.(n)}>{n}</button>
                </React.Fragment>
              ))}
            </div>
          )}
          {p.snippet_html && (
            <p className="mt-2 text-xs text-slate-400 leading-relaxed" onClick={click}
               dangerouslySetInnerHTML={{ __html: p.snippet_html }} />
          )}
          <div className="mt-2 text-xs text-slate-500">
            {p.journal && <span>{p.journal} · </span>}
            {p.doi && <a className="text-sky-400 hover:underline" href={`https://doi.org/${p.doi}`} target="_blank" rel="noreferrer">DOI:{p.doi}</a>}
          </div>
        </li>
      ))}
    </ol>
  )
}
