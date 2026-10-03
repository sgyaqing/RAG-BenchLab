import { request } from './client'

export interface RagSystem {
  id: number
  name: string
  platform: string
  base_url: string
  api_key: string | null
  headers: string
  body_template: string
  answer_path: string
  contexts_path: string | null
  llm_config_id: number | null
  llm_name: string | null
  platform_hint: string | null
  status: 'configuring' | 'completed' | 'failed'
  progress: number
  stage: string
  error: string | null
  created_at: string
  completed_at: string | null
}

export interface RagSystemPage {
  total: number
  items: RagSystem[]
}

export interface RagSystemCreateInput {
  mode: 'smart' | 'manual'
  name: string
  base_url: string
  api_key?: string | null
  lang?: string
  // smart
  llm_config_id?: number
  platform_hint?: string
  // manual
  headers?: string
  body_template?: string
  answer_path?: string
  contexts_path?: string | null
}

export interface RagSystemUpdateInput {
  name: string
  base_url: string
  api_key?: string | null
  lang?: string
  headers: string
  body_template: string
  answer_path: string
  contexts_path?: string | null
}

export type RagSystemLogEntry = import('@/utils/log').LogEntry

export interface RagSystemLog {
  entries: RagSystemLogEntry[]
  error: string | null
}

export function listRagSystems(name: string, page: number, pageSize: number) {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
  if (name) params.set('name', name)
  return request<RagSystemPage>(`/api/rag-systems?${params}`)
}

export function checkRagSystemName(name: string, excludeId?: number) {
  const params = new URLSearchParams({ name })
  if (excludeId) params.set('exclude_id', String(excludeId))
  return request<{ available: boolean }>(`/api/rag-systems/check-name?${params}`)
}

export function createRagSystem(input: RagSystemCreateInput) {
  return request<RagSystem>('/api/rag-systems', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export function updateRagSystem(id: number, input: RagSystemUpdateInput) {
  return request<RagSystem>(`/api/rag-systems/${id}`, {
    method: 'PUT',
    body: JSON.stringify(input),
  })
}

export function deleteRagSystem(id: number) {
  return request<void>(`/api/rag-systems/${id}`, { method: 'DELETE' })
}

export function getRagSystemLog(id: number) {
  return request<RagSystemLog>(`/api/rag-systems/${id}/log`)
}
