import { describe, expect, it } from 'vitest'
import { evalLogEntryText, evalStageText } from '../src/components/EvaluationPage'
import type { EvalRun } from '../src/api/evaluation'
import zh from '../src/i18n/locales/zh'

// Minimal t(): interpolate {placeholder} from the zh locale table.
function t(key: string, params?: Record<string, string | number>): string {
  const [ns, k] = key.split('.')
  const table = (zh as Record<string, Record<string, string>>)[ns]
  let text = table?.[k] ?? key
  for (const [p, v] of Object.entries(params ?? {})) {
    text = text.split(`{${p}}`).join(String(v))
  }
  return text
}

function runWith(partial: Partial<EvalRun>): EvalRun {
  return {
    id: 1,
    name: 'r',
    testset_id: 1,
    testset_name: 'ts',
    rag_system_id: 1,
    rag_system_name: 'rag',
    run_base_url: 'http://rag.local/query',
    timeout: 120,
    llm_name: 'l',
    embedding_name: 'e',
    concurrency: 4,
    judge_concurrency: 16,
    metrics: '[]',
    has_contexts: true,
    status: 'running',
    progress: 0,
    stage: 'init',
    stage_done: 0,
    stage_total: 0,
    total_items: 0,
    failed_items: 0,
    summary: '{}',
    error: null,
    created_at: '',
    completed_at: null,
    ...partial,
  }
}

describe('evalStageText', () => {
  it('maps known stages with counters', () => {
    expect(evalStageText(t, runWith({ stage: 'querying', stage_done: 3, stage_total: 10 }))).toBe(
      '正在调用RAG系统（3/10）',
    )
    expect(evalStageText(t, runWith({ stage: 'evaluating', stage_done: 2, stage_total: 4 }))).toBe(
      '正在计算测试指标（2/4）',
    )
  })

  it('never shows the internal stage code to the user', () => {
    // A stale cached bundle can lack a newly added stage code; showing
    // "future_stage" in the progress bar would be meaningless to a user.
    expect(evalStageText(t, runWith({ stage: 'future_stage' }))).toBe('处理中…')
  })

  it('shows terminal states', () => {
    expect(evalStageText(t, runWith({ status: 'completed' }))).toBe('已完成')
    expect(evalStageText(t, runWith({ status: 'failed' }))).toBe('失败')
  })
})

describe('evalLogEntryText', () => {
  it('interpolates params and drops nulls', () => {
    const text = evalLogEntryText(t, {
      time: '',
      key: 'queryDone',
      params: { total: 30, failed: 1, seconds: 12.5 },
    })
    expect(text).toBe('被测系统调用完成：共30条，失败1条，耗时12.5秒')
  })
})
