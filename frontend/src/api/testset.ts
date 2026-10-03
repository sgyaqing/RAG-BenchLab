import { request } from './client'

export interface Testset {
  id: number
  name: string
  corpus_id: number
  corpus_name: string
  reuse_kg: boolean
  llm_name: string
  embedding_name: string
  llm_concurrency: number
  llm_max_tokens: number
  n_single: number
  n_multi_specific: number
  n_multi_abstract: number
  status: 'generating' | 'completed' | 'failed'
  progress: number
  stage: string
  stage_done: number
  stage_total: number
  actual_single: number
  actual_multi_specific: number
  actual_multi_abstract: number
  error: string | null
  created_at: string
  completed_at: string | null
  edited: boolean
  item_count: number
}

export interface TestsetPage {
  total: number
  items: Testset[]
}

export type TestsetLogEntry = import('@/utils/log').LogEntry

export interface TestsetLog {
  entries: TestsetLogEntry[]
  token_usage: Record<string, number>
  error: string | null
}

export interface TestsetItem {
  id: number
  seq: number
  user_input: string
  reference: string
  reference_contexts: string[]
  synthesizer_name: string
  persona_name: string
  edited: boolean
  has_original: boolean
}

export interface TestsetItemPage {
  total: number
  items: TestsetItem[]
}

export interface TestsetCreateInput {
  name: string
  corpus_id: number
  reuse_kg: boolean
  llm_config_id: number
  embedding_config_id: number
  llm_concurrency: number
  llm_max_tokens: number
  n_single: number
  n_multi_specific: number
  n_multi_abstract: number
  prompt_language: 'auto' | 'zh' | 'en'
  amplify: number
  gen_amplify: number
}

export function listTestsets(name: string, page: number, pageSize: number) {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
  if (name) params.set('name', name)
  return request<TestsetPage>(`/api/testsets?${params}`)
}

export function checkTestsetName(name: string) {
  return request<{ available: boolean }>(
    `/api/testsets/check-name?name=${encodeURIComponent(name)}`,
  )
}

/** Import a testset from a JSONL file (multipart; not JSON — no `request()`). */
export async function importTestset(name: string, file: File): Promise<Testset> {
  const formData = new FormData()
  formData.append('name', name)
  formData.append('file', file)
  const resp = await fetch('/api/testsets/import', { method: 'POST', body: formData })
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
  return resp.json() as Promise<Testset>
}

export function createTestset(input: TestsetCreateInput) {
  return request<Testset>('/api/testsets', { method: 'POST', body: JSON.stringify(input) })
}

export function getTestset(id: number) {
  return request<Testset>(`/api/testsets/${id}`)
}

export function getTestsetLog(id: number) {
  return request<TestsetLog>(`/api/testsets/${id}/log`)
}

export function deleteTestset(id: number) {
  return request<void>(`/api/testsets/${id}`, { method: 'DELETE' })
}

export function resumeTestset(id: number) {
  return request<Testset>(`/api/testsets/${id}/resume`, { method: 'POST' })
}

export function listTestsetItems(id: number, page: number, pageSize: number) {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
  return request<TestsetItemPage>(`/api/testsets/${id}/items?${params}`)
}

export function updateTestsetItem(id: number, itemId: number, userInput: string, reference: string) {
  return request<TestsetItem>(`/api/testsets/${id}/items/${itemId}`, {
    method: 'PUT',
    body: JSON.stringify({ user_input: userInput, reference }),
  })
}

export function restoreTestsetItem(id: number, itemId: number) {
  return request<TestsetItem>(`/api/testsets/${id}/items/${itemId}/restore`, { method: 'POST' })
}

export function deleteTestsetItem(id: number, itemId: number) {
  return request<void>(`/api/testsets/${id}/items/${itemId}`, { method: 'DELETE' })
}

/** "30(10/10/10)" summary when every item's type is known; bare total otherwise
 *  (imported testsets may carry no/unknown synthesizer types). */
export function formatQaCount(
  t: Pick<
    Testset,
    'actual_single' | 'actual_multi_specific' | 'actual_multi_abstract' | 'item_count'
  >,
): string {
  const sum = t.actual_single + t.actual_multi_specific + t.actual_multi_abstract
  if (sum === t.item_count) {
    return `${sum}(${t.actual_single}/${t.actual_multi_specific}/${t.actual_multi_abstract})`
  }
  return `${t.item_count}`
}

/** reference_contexts summary: "3 段上下文 · 首段预览" handled by i18n at call site. */
export function firstContextPreview(contexts: string[], maxLen = 40): string {
  const first = contexts[0] ?? ''
  return first.length > maxLen ? first.slice(0, maxLen) + '…' : first
}
