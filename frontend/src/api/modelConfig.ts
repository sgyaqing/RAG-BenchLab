import { request } from './client'

export interface ModelConfig {
  id: number
  name: string
  type: 'llm' | 'embedding'
  api_format: 'openai' | 'anthropic'
  base_url: string
  api_key: string | null
  model: string
  // "already_off" / "disabled" mean thinking is off (the list reads 是),
  // "unsupported" means it could not be turned off (否), null means no verdict.
  thinking_state: string | null
}

export interface ModelConfigInput {
  name: string
  type: 'llm' | 'embedding'
  api_format: 'openai' | 'anthropic'
  base_url: string
  api_key?: string | null
  model: string
  // The dialog's "add the setting that turns thinking off" box. Unticked means
  // the backend clears any stored setting rather than probing again.
  auto_disable_thinking?: boolean
}

export interface ModelConfigPage {
  total: number
  items: ModelConfig[]
}

export interface ConnectivityTestInput {
  type: 'llm' | 'embedding'
  api_format: 'openai' | 'anthropic'
  base_url: string
  api_key?: string | null
  model: string
}

export function listModelConfigs(name: string, page: number, pageSize: number) {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
  if (name) params.set('name', name)
  return request<ModelConfigPage>(`/api/model-configs?${params}`)
}

export function checkModelConfigName(name: string, excludeId?: number) {
  const params = new URLSearchParams({ name })
  // Editing keeps the record's own name, so it has to be excluded from the
  // check or every save would report it as taken.
  if (excludeId !== undefined) params.set('exclude_id', String(excludeId))
  return request<{ available: boolean }>(`/api/model-configs/check-name?${params}`)
}

export function createModelConfig(input: ModelConfigInput) {
  return request<ModelConfig>('/api/model-configs', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export function updateModelConfig(id: number, input: ModelConfigInput) {
  return request<ModelConfig>(`/api/model-configs/${id}`, {
    method: 'PUT',
    body: JSON.stringify(input),
  })
}

export function deleteModelConfig(id: number) {
  return request<void>(`/api/model-configs/${id}`, { method: 'DELETE' })
}

export function testConnectivity(input: ConnectivityTestInput) {
  return request<{ success: boolean; message: string; duration_ms: number | null }>(
    '/api/model-configs/test',
    {
      method: 'POST',
      body: JSON.stringify(input),
    },
  )
}

export function fetchAvailableModels(input: {
  api_format: 'openai' | 'anthropic'
  base_url: string
  api_key?: string | null
}) {
  return request<{ models: string[] }>('/api/model-configs/available-models', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}
