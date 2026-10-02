import React, { useEffect, useMemo, useState } from 'react'
import { api } from './api.js'
import GraphCanvas from './components/GraphCanvas.jsx'
import PaperList from './components/PaperList.jsx'
import ChatSidebar from './components/ChatSidebar.jsx'
import BridgePanel from './components/BridgePanel.jsx'
import IngestPanel from './components/IngestPanel.jsx'

const TABS = ['Papers', 'Bridge', 'Ingest']

export default function App() {
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [result, setResult] = useState(null)
  const [graph, setGraph] = useState({ nodes: [], links: [] })
  const [seed, setSeed] = useState(null)
  const [focus, setFocus] = useState(null)
  const [tab, setTab] = useState('Papers')
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState(null)
  const [edgeInfo, setEdgeInfo] = useState(null)

  useEffect(() => { api.stats().then(setStats).catch((e) => setErr(`Backend unreachable: ${e.message}`)) }, [])

  useEffect(() => {
    if (query.trim().length < 2) { setSuggestions([]); return }
    const t = setTimeout(() => api.suggest(query).then((r) => setSuggestions(r.suggestions)).catch(() => {}), 250)
    return () => clearTimeout(t)
  }, [query])

  const run = async (q) => {
    setErr(''); setLoading(true); setSuggestions([]); setEdgeInfo(null)
    try {
      const r = await api.search(q)
      setResult(r)
      if (r.found) { setGraph({ nodes: r.nodes, links: r.links }); setSeed(r.seed); setFocus(r.seed); setQuery(r.seed); setTab('Papers') }
      else { setGraph({ nodes: [], links: [] }); setErr(`No entity matches “${q}”.`) }
    } catch (e) { setErr(e.message) } finally { setLoading(false) }
  }

  const onLink = async (l) => {
    const s = l.source.id || l.source, t = l.target.id || l.target
    if (l.predicted) { setEdgeInfo({ title: `${s} ↔ ${t}`, predicted: true, score: l.confidence, method: l.method }); return }
    try { const e = await api.edge(s, t); setEdgeInfo({ title: `${s} — ${t}`, ...e }) } catch (x) { setErr(x.message) }
  }

  const highlight = useMemo(() => new Set(focus ? [focus] : []), [focus])

  return (
    <div className="h-screen flex flex-col bg-slate-950 text-slate-200">
      <header className="p-3 border-b border-slate-800 relative">
        <form onSubmit={(e) => { e.preventDefault(); run(query) }} className="flex gap-2 items-center">
          <span className="font-bold text-sky-400 mr-2">LinkForge</span>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder='Search a concept, e.g. "chocolate"'
                 className="flex-1 rounded bg-slate-900 border border-slate-700 px-3 py-2 outline-none focus:border-sky-500" />
          <button className="rounded bg-sky-600 hover:bg-sky-500 px-4 py-2">{loading ? '…' : 'Search'}</button>
          {stats && <span className="text-xs text-slate-500 hidden md:block">{stats.graph.entities} entities · {stats.graph.relations} relations · {stats.graph.papers} papers · {stats.link_prediction?.method}</span>}
        </form>
        {suggestions.length > 0 && (
          <ul className="absolute z-10 left-24 right-40 mt-1 rounded border border-slate-700 bg-slate-900 shadow-lg">
            {suggestions.map((s) => (
              <li key={s.name}><button className="w-full text-left px-3 py-1.5 hover:bg-slate-800" onClick={() => run(s.name)}>
                {s.name} <span className="text-xs text-slate-500">{s.type}</span></button></li>
            ))}
          </ul>
        )}
        {err && <div className="mt-2 text-sm text-red-400">{err}</div>}
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_380px] min-h-0">
        <section className="flex flex-col min-h-0">
          <div className="relative flex-1 min-h-[300px]">
            {graph.nodes.length ? (
              <GraphCanvas nodes={graph.nodes} links={graph.links} seed={seed} highlight={highlight}
                           onNodeClick={(n) => { setFocus(n.id); }} onLinkClick={onLink} />
            ) : (
              <div className="h-full grid place-items-center text-slate-600 text-sm">Search a concept to explore its knowledge graph.</div>
            )}
            <div className="absolute bottom-2 left-3 text-xs text-slate-500 pointer-events-none">
              solid = extracted from papers · <span className="text-fuchsia-300">dashed = predicted (unverified)</span> · click a node to focus the chat, a line for evidence
            </div>
            {focus && <div className="absolute top-2 left-3 flex gap-2 items-center text-xs">
              <span className="rounded bg-slate-800 px-2 py-1">Focus: {focus}</span>
              <button className="rounded bg-slate-800 hover:bg-slate-700 px-2 py-1" onClick={() => run(focus)}>Expand</button>
            </div>}
            {edgeInfo && (
              <div className="absolute top-2 right-2 w-80 max-h-[70%] overflow-auto rounded border border-slate-700 bg-slate-900/95 p-3 text-xs">
                <div className="flex justify-between"><b>{edgeInfo.title}</b><button onClick={() => setEdgeInfo(null)}>✕</button></div>
                {edgeInfo.predicted ? <p className="mt-1 text-fuchsia-300">Predicted link ({edgeInfo.method}) · score {Math.round((edgeInfo.score || 0) * 100)}% — hypothesis, not a published finding.</p> : <>
                  {edgeInfo.evidence?.map((x, i) => <p key={i} className="mt-1 text-slate-300">“{x.text}” <span className="text-slate-500">{x.metadata?.doi && `DOI:${x.metadata.doi}`}</span></p>)}
                  {edgeInfo.papers?.map((p) => <p key={p.paper_id} className="mt-1 text-slate-500">{p.title} ({p.year})</p>)}
                </>}
              </div>
            )}
          </div>
          <div className="h-64 border-t border-slate-800 flex flex-col min-h-0">
            <nav className="flex border-b border-slate-800 text-sm">
              {TABS.map((t) => <button key={t} onClick={() => setTab(t)} className={`px-4 py-2 ${tab === t ? 'text-sky-400 border-b-2 border-sky-400' : 'text-slate-500'}`}>{t}</button>)}
            </nav>
            <div className="flex-1 overflow-y-auto">
              {tab === 'Papers' && <PaperList papers={result?.papers} onEntity={run} />}
              {tab === 'Bridge' && <BridgePanel onResult={(r) => { if (r.found) { setGraph({ nodes: r.nodes, links: r.links }); setSeed(r.source); setFocus(r.source) } }} />}
              {tab === 'Ingest' && <IngestPanel />}
            </div>
          </div>
        </section>
        <aside className="border-l border-slate-800 min-h-0 h-[50vh] lg:h-auto"><ChatSidebar focus={focus} /></aside>
      </main>
    </div>
  )
}
