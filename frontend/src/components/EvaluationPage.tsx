import { useCallback, useEffect, useRef, useState } from 'react'
import { App as AntdApp, Button, Input, Popconfirm, Progress, Table, Tag } from 'antd'
import {
  DeleteOutlined,
  FileSearchOutlined,
  FileTextOutlined,
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useLocation, useNavigate } from 'react-router-dom'
import { useI18n, type TFunc, type TParams } from '@/i18n'
import {
  deleteEvaluation,
  getEvaluationLog,
  listEvaluations,
  metricDisplayName,
  resumeEvaluation,
  type EvalLog,
  type EvalLogEntry,
  type EvalRun,
} from '@/api/evaluation'
import CreateEvalModal from './CreateEvalModal'
import LogModal from './LogModal'
import styles from './CorpusPage.module.css'

const POLL_INTERVAL_MS = 1500

function formatDateTime(iso: string | null): string {
  if (!iso) return ''
  // Backend stores UTC without a tz marker; parse as UTC, display local.
  const d = new Date(/[zZ]|[+-]\d{2}:?\d{2}$/.test(iso) ? iso : iso + 'Z')
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

/** Progress text for a run: the terminal states, else the stage code translated.
 * A stage the locale table does not know falls back to a generic phrase rather
 * than showing the internal code. */
export function evalStageText(t: TFunc, run: EvalRun): string {
  if (run.status === 'completed') return t('evaluation.statusCompleted')
  if (run.status === 'failed') return t('evaluation.statusFailed')
  const key = `evaluation.stage${run.stage[0].toUpperCase()}${run.stage.slice(1)}`
  const text = t(key, { done: run.stage_done, total: run.stage_total })
  return text === key ? t('common.processing') : text
}

/** One log line, or null when the locale table has no copy for this key — a
 * key the running bundle predates, which should not be shown as raw text. */
export function evalLogEntryText(t: TFunc, entry: EvalLogEntry): string | null {
  if (entry.key === 'judgeCache' && !entry.params.readEnabled) return t('evaluation.logJudgeCacheOff')
  const key = `evaluation.log${entry.key[0].toUpperCase()}${entry.key.slice(1)}`
  const params = Object.fromEntries(
    Object.entries(entry.params).filter(([, v]) => v != null),
  ) as TParams
  const text = t(key, params)
  return text === key ? null : text
}

export default function EvaluationPage() {
  const { t } = useI18n()
  const { notification } = AntdApp.useApp()
  const navigate = useNavigate()
  const location = useLocation()
  // The report page sends the list position back, so returning from a run
  // lands on the page the user left.
  const navState = location.state as
    | { page: number; pageSize: number; searchName: string }
    | null
    | undefined

  const [data, setData] = useState<EvalRun[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(navState?.page ?? 1)
  const [pageSize, setPageSize] = useState(navState?.pageSize ?? 10)
  const [searchName, setSearchName] = useState(navState?.searchName ?? '')
  const [loading, setLoading] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [logTarget, setLogTarget] = useState<EvalRun | null>(null)
  const [logData, setLogData] = useState<EvalLog | null>(null)

  // Typing in the search box fires a request per keystroke; without this a slow
  // earlier response would land last and show results for a stale keyword.
  const loadSeq = useRef(0)

  const load = useCallback(async () => {
    const seq = ++loadSeq.current
    setLoading(true)
    try {
      const res = await listEvaluations(searchName, page, pageSize)
      if (seq !== loadSeq.current) return
      setData(res.items)
      setTotal(res.total)
    } finally {
      if (seq === loadSeq.current) setLoading(false)
    }
  }, [searchName, page, pageSize])

  useEffect(() => {
    void load()
  }, [load])

  const hasRunning = data.some((r) => r.status === 'running')
  useEffect(() => {
    if (!hasRunning) return
    const timer = setInterval(() => void load(), POLL_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [hasRunning, load])

  async function openLog(record: EvalRun) {
    setLogTarget(record)
    setLogData(null)
    try {
      setLogData(await getEvaluationLog(record.id))
    } catch (e) {
      // without this the dialog opens and stays blank, looking frozen
      notification.error({
        message: t('common.loadFailed'),
        description: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
      setLogTarget(null)
    }
  }

  async function onResume(record: EvalRun) {
    try {
      await resumeEvaluation(record.id)
      notification.success({
        message: t('evaluation.resumeStarted'),
        placement: 'topRight',
        duration: 3,
      })
      void load()
    } catch (e) {
      notification.error({
        message: t('common.saveFailed'),
        description: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
    }
  }

  async function onDelete(record: EvalRun) {
    try {
      await deleteEvaluation(record.id)
      notification.success({
        message: t('evaluation.deleteSuccess'),
        placement: 'topRight',
        duration: 3,
      })
      void load()
    } catch (e) {
      notification.error({
        message: t('common.deleteFailed'),
        description: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
    }
  }

  const columns: ColumnsType<EvalRun> = [
    { title: t('evaluation.colName'), dataIndex: 'name', key: 'name' },
    { title: t('evaluation.colTestset'), dataIndex: 'testset_name', key: 'testset' },
    { title: t('evaluation.colRagSystem'), dataIndex: 'rag_system_name', key: 'ragSystem' },
    {
      title: t('evaluation.colStatus'),
      key: 'status',
      width: 260,
      render: (_value, record) =>
        record.status === 'completed' ? (
          <Tag color="success">{t('evaluation.statusCompleted')}</Tag>
        ) : record.status === 'failed' ? (
          <Tag color="error">{t('evaluation.statusFailed')}</Tag>
        ) : (
          <span className={styles.progressWrap}>
            <span className={styles.progressCell}>
              <Progress
                percent={Math.round(record.progress)}
                size="small"
                showInfo={false}
                status="active"
                strokeColor="#1677ff"
              />
              <span className={styles.progressText}>{Math.round(record.progress)}%</span>
            </span>
            <span className={styles.stageText}>{evalStageText(t, record)}</span>
          </span>
        ),
    },
    {
      title: t('evaluation.colScores'),
      key: 'scores',
      render: (_value, record) => {
        if (record.status !== 'completed') return null
        const summary = JSON.parse(record.summary || '{}') as Record<string, number>
        return (
          <span style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'flex-start' }}>
            {Object.entries(summary).map(([key, value]) => (
              <Tag key={key}>{`${metricDisplayName(key)}: ${value}`}</Tag>
            ))}
            {record.failed_items > 0 && (
              <Tag color="warning">
                {t('evaluation.failedItemsTag', { count: record.failed_items })}
              </Tag>
            )}
          </span>
        )
      },
    },
    {
      title: t('evaluation.colCompletedAt'),
      key: 'completedAt',
      render: (_value, record) => formatDateTime(record.completed_at),
    },
    {
      title: t('evaluation.colActions'),
      key: 'actions',
      render: (_value, record) =>
        record.status === 'running' ? null : (
          <span>
            {record.status === 'completed' && (
              <Button
                type="link"
                size="small"
                icon={<FileSearchOutlined />}
                onClick={() =>
                  navigate(`/evaluations/${record.id}`, {
                    state: { page, pageSize, searchName },
                  })
                }
              >
                {t('evaluation.report')}
              </Button>
            )}
            <Button
              type="link"
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => void openLog(record)}
            >
              {t('evaluation.log')}
            </Button>
            {record.status === 'failed' && (
              <Button
                type="link"
                size="small"
                icon={<ReloadOutlined />}
                onClick={() => void onResume(record)}
              >
                {t('evaluation.resume')}
              </Button>
            )}
            <Popconfirm
              title={t('evaluation.deleteConfirm')}
              onConfirm={() => void onDelete(record)}
            >
              <Button type="link" size="small" danger icon={<DeleteOutlined />}>
                {t('evaluation.delete')}
              </Button>
            </Popconfirm>
          </span>
        ),
    },
  ]

  return (
    <main className={styles.page}>
      <div className={styles.toolbar}>
        <Input
          allowClear
          defaultValue={searchName}
          prefix={<SearchOutlined />}
          placeholder={t('evaluation.searchPlaceholder')}
          style={{ width: 280 }}
          onChange={(e) => {
            setPage(1)
            setSearchName(e.target.value)
          }}
        />
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateOpen(true)}>
          {t('evaluation.createTest')}
        </Button>
      </div>

      <Table
        rowKey="id"
        loading={loading && data.length === 0}
        columns={columns}
        dataSource={data}
        pagination={{
          current: page,
          pageSize,
          total,
          showSizeChanger: true,
          pageSizeOptions: [10, 20, 50],
          showTotal: (count, range) =>
            t('modelConfig.paginationSummary', {
              total: count,
              from: range[0],
              to: range[1],
            }),
          onChange: (nextPage, nextPageSize) => {
            setPage(nextPage)
            setPageSize(nextPageSize)
          },
        }}
      />

      <CreateEvalModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={() => {
          setCreateOpen(false)
          void load()
        }}
      />

      <LogModal
        title={t('evaluation.logTitle')}
        open={logTarget !== null}
        onClose={() => setLogTarget(null)}
        entries={logData?.entries ?? []}
        error={logData?.error}
        textOf={(entry) => evalLogEntryText(t, entry)}
      />
    </main>
  )
}
