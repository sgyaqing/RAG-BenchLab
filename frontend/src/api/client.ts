export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`
    try {
      const body = await resp.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // keep the HTTP status as the message
    }
    throw new Error(detail)
  }
  if (resp.status === 204) return undefined as T
  return resp.json() as Promise<T>
}
