const BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

// Admin key lives only in memory (never baked into the bundle): typed into the Ingest tab.
let adminKey = ''
export const setAdminKey = (k) => { adminKey = k }

async function req(path, opts = {}) {
  const headers = { ...(opts.headers || {}), ...(adminKey ? { 'X-API-Key': adminKey } : {}) }
  const res = await fetch(`${BASE}${path}`, { ...opts, headers })
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail || detail } catch { /* ignore */ }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}
const qs = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v !== undefined && v !== '')).toString()

export const api = {
  search: (q, topK = 20, hops = 3) => req(`/api/search?${qs({ q, top_k: topK, hops })}`),
  suggest: (q) => req(`/api/search/suggest?${qs({ q })}`),
  bridge: (a, b, maxHops = 6) => req(`/api/search/bridge?${qs({ a, b, max_hops: maxHops })}`),
  stats: () => req('/api/graph/stats'),
  edge: (source, target) => req(`/api/graph/edge?${qs({ source, target })}`),
  chat: (question, focus, history) =>
    req('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                       body: JSON.stringify({ question, focus, history }) }),
  ingestQuery: (query, source = 'europe_pmc', limit = 10) =>
    req('/api/ingest/query', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                               body: JSON.stringify({ query, source, limit }) }),
  ingestUpload: (file) => { const f = new FormData(); f.append('file', file)
    return req('/api/ingest/upload', { method: 'POST', body: f }) },
  job: (id) => req(`/api/ingest/jobs/${id}`),
}
