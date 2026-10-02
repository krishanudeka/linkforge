import React, { useState, useEffect, useRef } from 'react'
import { api, setAdminKey } from '../api.js'

export default function IngestPanel() {
  const [q, setQ] = useState('')
  const [source, setSource] = useState('europe_pmc')
  const [limit, setLimit] = useState(5)
  const [job, setJob] = useState(null)
  const [err, setErr] = useState('')
  const timer = useRef()

  const poll = (id) => {
    clearInterval(timer.current)
    timer.current = setInterval(async () => {
      try {
        const j = await api.job(id); setJob(j)
        if (j.status === 'done' || j.status === 'failed') clearInterval(timer.current)
      } catch (e) { setErr(e.message); clearInterval(timer.current) }
    }, 2000)
  }
  useEffect(() => () => clearInterval(timer.current), [])

  const start = async (fn) => {
    setErr(''); setJob(null)
    try { const { job_id } = await fn(); setJob({ id: job_id, status: 'queued', log: [] }); poll(job_id) } catch (e) { setErr(e.message) }
  }

  return (
    <div className="p-3 text-sm space-y-4">
      <input type="password" placeholder="Admin API key (if the server requires one)" onChange={(e) => setAdminKey(e.target.value)}
             className="w-full rounded bg-slate-900 border border-slate-700 px-2 py-1.5 text-xs" />
      <form onSubmit={(e) => { e.preventDefault(); start(() => api.ingestQuery(q, source, Number(limit))) }} className="space-y-2">
        <div className="font-medium text-slate-200">Scrape open-access papers</div>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. curcumin alzheimer" className="w-full rounded bg-slate-900 border border-slate-700 px-2 py-1.5" />
        <div className="flex gap-2">
          <select value={source} onChange={(e) => setSource(e.target.value)} className="rounded bg-slate-900 border border-slate-700 px-2 py-1.5">
            <option value="europe_pmc">Europe PMC</option><option value="pubmed">PubMed</option><option value="semantic_scholar">Semantic Scholar</option>
          </select>
          <input type="number" min="1" max="50" value={limit} onChange={(e) => setLimit(e.target.value)} className="w-20 rounded bg-slate-900 border border-slate-700 px-2 py-1.5" />
          <button className="rounded bg-sky-600 hover:bg-sky-500 px-3">Ingest</button>
        </div>
      </form>
      <div>
        <div className="font-medium text-slate-200 mb-1">Or upload a PDF</div>
        <input type="file" accept="application/pdf" onChange={(e) => e.target.files[0] && start(() => api.ingestUpload(e.target.files[0]))} className="text-xs" />
      </div>
      {err && <p className="text-red-400">{err}</p>}
      {job && (
        <div className="rounded border border-slate-800 p-2 text-xs">
          <div>Job <code>{job.id}</code>: <b className={job.status === 'failed' ? 'text-red-400' : job.status === 'done' ? 'text-emerald-400' : 'text-amber-300'}>{job.status}</b></div>
          {job.error && <div className="text-red-400">{job.error}</div>}
          {job.result?.by_status && <div>Papers: {JSON.stringify(job.result.by_status)} · triplets: {job.result.triplets}</div>}
          {job.result?.status && <div>Result: {job.result.status} · triplets: {job.result.triplets}</div>}
          <pre className="mt-1 max-h-40 overflow-auto text-slate-500 whitespace-pre-wrap">{(job.log || []).slice(-12).join('\n')}</pre>
        </div>
      )}
    </div>
  )
}
