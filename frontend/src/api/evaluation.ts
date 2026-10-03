import { request } from './client'

// Metric keys are the internal/ragas keys; display names are the standard
// ragas names (response_relevancy is now called Response Relevancy upstream).
export const METRIC_KEYS = [
  'context_precision',
  'context_recall',
  'faithfulness',
  'response_relevancy',
  'answer_correctness',
] as const

export const METRIC_DISPLAY: Record<string, string> = {
  context_precision: 'Context Precision',
  context_recall: 'Context Recall',
  faithfulness: 'Faithfulness',
  response_relevancy: 'Response Relevancy',
  answer_correctness: 'Answer Correctness',
}

export function metricDisplayName(key: string): string {
  return METRIC_DISPLAY[key] ?? key
}

export interface EvalRun {
  id: number
  name: string
  testset_id: number
  testset_name: string
  rag_system_id: number
  rag_system_name: string
  run_base_url: string
  timeout: number
  llm_name: string
  embedding_name: string
  concurrency: number
  judge_concurrency: number
  metrics: string // JSON array
  has_contexts: boolean
  status: 'running' | 'completed' | 'failed'
  progress: number
  stage: string
  stage_done: number
  stage_total: number
  total_items: number
  failed_items: number
  summary: string // JSON object {metric: avg}
  error: string | null
  created_at: string
  completed_at: string | null
}

export interface EvalRunPage {
  total: number
  items: EvalRun[]
}

export type EvalLogEntry = import('@/utils/log').LogEntry

export interface EvalLog {
  entries: EvalLogEntry[]
  token_usage: { prompt?: number; completion?: number; calls?: number }
  error: string | null
}

export interface EvalItem {
  id: number
  seq: number
  user_input: string
  reference: string
  answer: string | null
  contexts: string // JSON array
  scores: string // JSON object
  error: string | null
}

export interface EvalItemPage {
  total: number
  items: EvalItem[]
}

export interface EvalRunCreateInput {
  name: string
  testset_id: number
  rag_system_id: number
  llm_config_id: number
  embedding_config_id: number
  concurrency: number
  judge_concurrency?: number
  use_judge_cache?: boolean
  metrics: string[]
  base_url?: string
  api_key?: string | null
  timeout?: number
}

export function listEvaluations(name: string, page: number, pageSize: number) {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
  if (name) params.set('name', name)
  return request<EvalRunPage>(`/api/evaluations?${params}`)
}

export function checkEvalName(name: string) {
  return request<{ available: boolean }>(
    `/api/evaluations/check-name?name=${encodeURIComponent(name)}`,
  )
}

export function createEvaluation(input: EvalRunCreateInput) {
  return request<EvalRun>('/api/evaluations', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export function getEvaluation(id: number) {
  return request<EvalRun>(`/api/evaluations/${id}`)
}

export function getEvaluationLog(id: number) {
  return request<EvalLog>(`/api/evaluations/${id}/log`)
}

export function listEvaluationItems(id: number, page: number, pageSize: number) {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
  return request<EvalItemPage>(`/api/evaluations/${id}/items?${params}`)
}

export function resumeEvaluation(id: number) {
  return request<EvalRun>(`/api/evaluations/${id}/resume`, { method: 'POST' })
}

export function deleteEvaluation(id: number) {
  return request<void>(`/api/evaluations/${id}`, { method: 'DELETE' })
}
