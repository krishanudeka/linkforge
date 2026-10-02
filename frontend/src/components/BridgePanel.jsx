import React, { useState } from 'react'
import { api } from '../api.js'

export default function BridgePanel({ onResult }) {
  const [a, setA] = useState('')
  const [b, setB] = useState('')
  const [res, setRes] = useState(null)
  const [err, setErr] = useState('')
  const go = async (e) => {
    e.preventDefault(); setErr(''); setRes(null)
    try { const r = await api.bridge(a, b); setRes(r); onResult?.(r) } catch (x) { setErr(x.message) }
  }
  return (
    <div className="p-3 text-sm">
      <form onSubmit={go} className="flex gap-2 mb-3">
        <input value={a} onChange={(e) => setA(e.target.value)} placeholder="Concept A (e.g. curcumin)" className="flex-1 min-w-0 rounded bg-slate-900 border border-slate-700 px-2 py-1.5" />
        <span className="self-center text-slate-500">→ ? →</span>
        <input value={b} onChange={(e) => setB(e.target.value)} placeholder="Concept D" className="flex-1 min-w-0 rounded bg-slate-900 border border-slate-700 px-2 py-1.5" />
        <button className="rounded bg-emerald-600 hover:bg-emerald-500 px-3">Find</button>
      </form>
      {err && <p className="text-red-400">{err}</p>}
      {res && !res.found && <p className="text-slate-400">No path found.
        {res.predicted_link && <> Predicted link score: <b className="text-fuchsia-300">{Math.round(res.predicted_link.score * 100)}%</b> ({res.predicted_link.method}, unverified)</>}</p>}
      {res?.paths?.map((p, i) => (
        <div key={i} className="mb-3 rounded border border-slate-800 p-2">
          <div className="text-xs text-slate-500 mb-1">Path {i + 1} · {p.hops} hops · confidence {Math.round(p.score * 100)}%</div>
          <div className="text-sky-300">{p.nodes.join(' → ')}</div>
          <ul className="mt-1 text-xs text-slate-400 space-y-1">
            {p.steps.map((s, k) => (
              <li key={k}>{s.source} <i>{s.type}</i> {s.target}{s.evidence?.[0] && <span className="text-slate-500"> — “{s.evidence[0].text}”</span>}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}
