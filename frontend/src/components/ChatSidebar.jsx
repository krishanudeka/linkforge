import React, { useState, useRef, useEffect } from 'react'
import { api } from '../api.js'

export default function ChatSidebar({ focus }) {
  const [msgs, setMsgs] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const end = useRef()
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  const send = async (e) => {
    e.preventDefault()
    const q = input.trim()
    if (!q || busy) return
    const history = msgs.map((m) => ({ role: m.role, content: m.content }))
    setMsgs((m) => [...m, { role: 'user', content: q }])
    setInput(''); setBusy(true)
    try {
      const r = await api.chat(q, focus ? [focus] : undefined, history)
      setMsgs((m) => [...m, { role: 'assistant', content: r.answer, ctx: r }])
    } catch (err) {
      setMsgs((m) => [...m, { role: 'assistant', content: `Error: ${err.message}` }])
    } finally { setBusy(false) }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-3 space-y-3 text-sm">
        {!msgs.length && <p className="text-slate-500">Ask about the graph, e.g. “Why is curcumin connected to Alzheimer’s disease?”{focus && <> Focus: <b className="text-slate-300">{focus}</b></>}</p>}
        {msgs.map((m, i) => (
          <div key={i} className={m.role === 'user' ? 'text-right' : ''}>
            <div className={`inline-block max-w-full text-left rounded-lg px-3 py-2 whitespace-pre-wrap ${m.role === 'user' ? 'bg-sky-700 text-white' : 'bg-slate-800 text-slate-200'}`}>
              {m.content}
            </div>
            {m.ctx?.predicted?.length > 0 && (
              <div className="mt-1 text-xs text-fuchsia-300">
                Predicted (unverified): {m.ctx.predicted.map((p) => `${p.source}↔${p.target} ${Math.round(p.score * 100)}%`).join(', ')}
              </div>
            )}
            {m.ctx?.excerpts?.length > 0 && (
              <details className="mt-1 text-xs text-slate-500">
                <summary className="cursor-pointer">Sources ({m.ctx.excerpts.length})</summary>
                <ul className="mt-1 space-y-1">
                  {m.ctx.excerpts.map((x, k) => (
                    <li key={k}>[S{k + 1}] {x.doi ? <a className="text-sky-400" href={`https://doi.org/${x.doi}`} target="_blank" rel="noreferrer">DOI:{x.doi}</a> : x.title} — {x.text.slice(0, 140)}…</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        ))}
        {busy && <div className="text-slate-500">Thinking…</div>}
        <div ref={end} />
      </div>
      <form onSubmit={send} className="p-3 border-t border-slate-800 flex gap-2">
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask a question about the graph..."
               className="flex-1 rounded bg-slate-900 border border-slate-700 px-3 py-2 text-sm outline-none focus:border-sky-500" />
        <button disabled={busy} className="rounded bg-sky-600 hover:bg-sky-500 disabled:opacity-50 px-3 text-sm">Send</button>
      </form>
    </div>
  )
}
