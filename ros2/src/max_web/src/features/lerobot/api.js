// Thin fetch helpers for the LeRobot Editor backend.
// All endpoints are served under /api/dataset/* and proxied to :8765 by Vite.

async function jsonOrThrow(res) {
  if (!res.ok) {
    let detail = ''
    try {
      const j = await res.json()
      detail = j.detail || JSON.stringify(j)
    } catch {
      detail = await res.text()
    }
    throw new Error(`${res.status} ${res.statusText}: ${detail}`)
  }
  return res.json()
}

export async function loadDataset(path) {
  const res = await fetch('/api/dataset/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  })
  return jsonOrThrow(res)
}

export async function listEpisodes() {
  return jsonOrThrow(await fetch('/api/dataset/episodes'))
}

export async function getEpisodeFrames(idx) {
  return jsonOrThrow(await fetch(`/api/dataset/episodes/${idx}/frames`))
}

export async function exportDataset(outPath, trims) {
  const res = await fetch('/api/dataset/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ out_path: outPath, trims }),
  })
  return jsonOrThrow(res)
}

export async function exportStatus() {
  return jsonOrThrow(await fetch('/api/dataset/export/status'))
}

