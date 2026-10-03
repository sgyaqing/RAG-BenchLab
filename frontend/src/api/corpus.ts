import { request } from './client'

export interface Corpus {
  id: number
  name: string
  status: 'converting' | 'completed'
  total_files: number
  processed_files: number
  success_files: number
  has_kg: boolean
  convert_seconds: number | null
  created_at: string
  completed_at: string | null
}

export interface CorpusPage {
  total: number
  items: Corpus[]
}

export interface CorpusLog {
  /** Timestamped events, as the other three logs carry. */
  entries: import('@/utils/log').LogEntry[]
  total_files: number
  success_files: number
  failed_files: string[]
  convert_seconds: number | null
  completed_at: string | null
}

export function listCorpora(name: string, page: number, pageSize: number) {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
  if (name) params.set('name', name)
  return request<CorpusPage>(`/api/corpora?${params}`)
}

export function checkCorpusName(name: string) {
  return request<{ available: boolean }>(
    `/api/corpora/check-name?name=${encodeURIComponent(name)}`,
  )
}

export function detectCorpusLanguage(corpusId: number) {
  return request<{ language: 'zh' | 'en' }>(`/api/corpora/${corpusId}/detect-language`)
}

export function createCorpus(name: string) {
  return request<Corpus>('/api/corpora', { method: 'POST', body: JSON.stringify({ name }) })
}

export function uploadCorpusFile(corpusId: number, file: File, path: string) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('path', path)
  return fetch(`/api/corpora/${corpusId}/files`, { method: 'POST', body: formData }).then(
    async (resp) => {
      if (!resp.ok) {
        let detail = `HTTP ${resp.status}`
        try {
          const body = await resp.json()
          if (typeof body.detail === 'string') detail = body.detail
        } catch {
          // keep HTTP status
        }
        throw new Error(detail)
      }
    },
  )
}

export function processCorpus(corpusId: number) {
  return request<Corpus>(`/api/corpora/${corpusId}/process`, { method: 'POST' })
}

export function getCorpusLog(corpusId: number) {
  return request<CorpusLog>(`/api/corpora/${corpusId}/log`)
}

export function deleteCorpus(corpusId: number) {
  return request<void>(`/api/corpora/${corpusId}`, { method: 'DELETE' })
}
