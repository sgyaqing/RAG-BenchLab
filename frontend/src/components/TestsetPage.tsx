import { useCallback, useEffect, useRef, useState } from 'react'
import {
  App as AntdApp,
  Button,
  Input,
  Popconfirm,
  Progress,
  Space,
  Table,
  Tag,
} from 'antd'
import {
  DeleteOutlined,
  DownloadOutlined,
  EditOutlined,
  FileTextOutlined,
  ImportOutlined,
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useNavigate, useLocation } from 'react-router-dom'
import { useI18n, type TFunc, type TParams } from '@/i18n'
import { downloadWithSaveAs } from '@/utils/download'
import LogModal from './LogModal'
import {
  deleteTestset,
  formatQaCount,
  getTestsetLog,
  listTestsets,
  resumeTestset,
  type Testset,
  type TestsetLog,
  type TestsetLogEntry,
} from '@/api/testset'
import CreateTestsetModal from './CreateTestsetModal'
import ImportTestsetModal from './ImportTestsetModal'
import styles from './CorpusPage.module.css'

const POLL_INTERVAL_MS = 1500

function formatDateTime(iso: string | null): string {
  if (!iso) return ''
  // Backend stores UTC without a tz marker; parse as UTC, display local.
  const d = new Date(/[zZ]|[+-]\d{2}:?\d{2}$/.test(iso) ? iso : iso + 'Z')
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

/** Map backend stage codes to i18n keys: building_kg -> stageBuildingKg. */
export function stageText(t: TFunc, ts: Testset): string {
  const camel = ts.stage
    .split('_')
    .map((w, i) => (i === 0 ? w : w[0].toUpperCase() + w.slice(1)))
    .join('')
  const key = `testset.stage${camel[0].toUpperCase()}${camel.slice(1)}`
  const text = t(key, { done: ts.stage_done, total: ts.stage_total })
  // A stale cached bundle can lack a newly added stage: never show the
  // internal code to the user.
  return text === key ? t('common.processing') : text
}

const TYPE_LABEL_KEYS: Record<string, string> = {
  single: 'testset.typeSingle',
  multi_specific: 'testset.typeMultiSpecific',
  multi_abstract: 'testset.typeMultiAbstract',
}

export function logEntryText(t: TFunc, entry: TestsetLogEntry): string | null {
  // Null-valued params would interpolate as "null"; same filter as utils/log.ts.
  const params: TParams = Object.fromEntries(
    Object.entries(entry.params).filter(([, v]) => v != null),
  ) as TParams
  if (typeof params.type === 'string' && TYPE_LABEL_KEYS[params.type]) {
    params.typeLabel = t(TYPE_LABEL_KEYS[params.type])
  }
  if (params.lang === 'zh') params.lang = t('testset.logLangZh')
  if (params.lang === 'en') params.lang = t('testset.logLangEn')
  const key = `testset.log${entry.key[0].toUpperCase()}${entry.key.slice(1)}`
  const text = t(key, params)
  // A stale cached bundle can lack a newly added log key: omit the line rather
  // than showing the internal key to the user.
  if (text === key) return null
  // Same reason, one step further. When the copy names a value an older entry
  // does not carry, interpolation leaves the placeholder standing and the line
  // would read "补生成2条（共{total}条）". An entry written before its line
  // gained a field cannot be rendered faithfully, so it is omitted.
  return /\{\w+\}/.test(text) ? null : text
}

export default function TestsetPage() {
  const { t } = useI18n()
  const { notification } = AntdApp.useApp()
  const navigate = useNavigate()
  const location = useLocation()

  // Restore list state (page/search) when returning from the edit page.
  const restore = location.state as
    | { page?: number; pageSize?: number; searchName?: string }
    | null

  const [data, setData] = useState<Testset[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(restore?.page ?? 1)
  const [pageSize, setPageSize] = useState(restore?.pageSize ?? 10)
  const [searchName, setSearchName] = useState(restore?.searchName ?? '')
  const [loading, setLoading] = useState(false)

  const [createOpen, setCreateOpen] = useState(false)
  const [importOpen, setImportOpen] = useState(false)
  const [logTestset, setLogTestset] = useState<Testset | null>(null)
  const [logData, setLogData] = useState<TestsetLog | null>(null)

  // Typing in the search box fires a request per keystroke; without this a slow
  // earlier response would land last and show results for a stale keyword.
  const loadSeq = useRef(0)

  const load = useCallback(async () => {
    const seq = ++loadSeq.current
    setLoading(true)
    try {
      const res = await listTestsets(searchName, page, pageSize)
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

  const hasGenerating = data.some((c) => c.status === 'generating')
  useEffect(() => {
    if (!hasGenerating) return
    const timer = setInterval(() => void load(), POLL_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [hasGenerating, load])

  async function onDelete(record: Testset) {
    try {
      await deleteTestset(record.id)
      notification.success({
        message: t('testset.deleteSuccess'),
        placement: 'topRight',
        duration: 3,
      })
      void load()
    } catch (e) {
      // a 409 (still generating) or 404 (removed in another tab) used to be a
      // silent no-op: the row stayed and nothing said why
      notification.error({
        message: t('common.deleteFailed'),
        description: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
    }
  }

  async function onExport(record: Testset) {
    try {
      await downloadWithSaveAs(`/api/testsets/${record.id}/export`, `${record.name}.jsonl`, {
        'application/x-ndjson': ['.jsonl'],
      })
    } catch {
      notification.error({
        message: t('testset.exportFailed'),
        placement: 'topRight',
        duration: 3,
      })
    }
  }

  async function openLog(record: Testset) {
    setLogTestset(record)
    setLogData(null)
    try {
      setLogData(await getTestsetLog(record.id))
    } catch (e) {
      // without this the dialog opens and stays blank, looking frozen
      notification.error({
        message: t('common.loadFailed'),
        description: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
      setLogTestset(null)
    }
  }

  async function onResume(record: Testset) {
    try {
      await resumeTestset(record.id)
      notification.success({
        message: t('testset.resumeStarted'),
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

  const columns: ColumnsType<Testset> = [
    { title: t('testset.colName'), dataIndex: 'name', key: 'name' },
    {
      title: t('testset.colQaCount'),
      key: 'qaCount',
      render: (_, r) => (r.status === 'completed' ? formatQaCount(r) : null),
    },
    {
      title: t('testset.colCorpus'),
      key: 'corpus_name',
      // corpus_id 0 is the import sentinel: no backing corpus, show a label.
      render: (_, r) => (r.corpus_id === 0 ? t('testset.importedCorpus') : r.corpus_name),
    },
    {
      title: t('testset.colStatus'),
      key: 'status',
      width: 260,
      render: (_, r) => {
        if (r.status === 'completed') {
          return <Tag color="success">{t('testset.statusCompleted')}</Tag>
        }
        if (r.status === 'failed') {
          return <Tag color="error">{t('testset.statusFailed')}</Tag>
        }
        return (
          <span className={styles.progressWrap}>
            <span className={styles.progressCell}>
              <Progress
                percent={Math.round(r.progress)}
                size="small"
                showInfo={false}
                status="active"
                strokeColor="#1677ff"
              />
              <span className={styles.progressText}>{Math.round(r.progress)}%</span>
            </span>
            <span className={styles.stageText}>{stageText(t, r)}</span>
          </span>
        )
      },
    },
    {
      title: t('testset.colCompletedAt'),
      key: 'completedAt',
      render: (_, r) => (
        <span>
          {formatDateTime(r.completed_at)}
          {r.edited && (
            <Tag color="warning" style={{ marginLeft: 6 }}>
              {t('testset.editedTag')}
            </Tag>
          )}
        </span>
      ),
    },
    {
      title: t('testset.colActions'),
      key: 'actions',
      render: (_, r) => {
        if (r.status === 'generating') return null
        return (
          <span>
            <Button
              type="link"
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => void openLog(r)}
            >
              {t('testset.log')}
            </Button>
            {r.status === 'completed' && (
              <Button
                type="link"
                size="small"
                icon={<EditOutlined />}
                onClick={() =>
                  navigate(`/testsets/${r.id}/edit`, {
                    state: { page, pageSize, searchName },
                  })
                }
              >
                {t('testset.edit')}
              </Button>
            )}
            {r.status === 'completed' && (
              <Button
                type="link"
                size="small"
                icon={<DownloadOutlined />}
                onClick={() => void onExport(r)}
              >
                {t('testset.export')}
              </Button>
            )}
            {r.status === 'failed' && (
              <Button
                type="link"
                size="small"
                icon={<ReloadOutlined />}
                onClick={() => void onResume(r)}
              >
                {t('testset.resume')}
              </Button>
            )}
            <Popconfirm
              title={t('testset.deleteConfirm')}
              onConfirm={() => void onDelete(r)}
            >
              <Button type="link" size="small" danger icon={<DeleteOutlined />}>
                {t('testset.delete')}
              </Button>
            </Popconfirm>
          </span>
        )
      },
    },
  ]

  return (
    <main className={styles.page}>
      <div className={styles.toolbar}>
        <Input
          allowClear
          defaultValue={searchName}
          prefix={<SearchOutlined />}
          placeholder={t('testset.searchPlaceholder')}
          style={{ width: 280 }}
          onChange={(e) => {
            setPage(1)
            setSearchName(e.target.value)
          }}
        />
        <Space>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateOpen(true)}>
            {t('testset.create')}
          </Button>
          <Button icon={<ImportOutlined />} onClick={() => setImportOpen(true)}>
            {t('testset.import')}
          </Button>
        </Space>
      </div>

      <Table<Testset>
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
          showTotal: (totalCount, range) =>
            t('modelConfig.paginationSummary', {
              total: totalCount,
              from: range[0],
              to: range[1],
            }),
          onChange: (p, ps) => {
            setPage(p)
            setPageSize(ps)
          },
        }}
      />

      <CreateTestsetModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={() => {
          setCreateOpen(false)
          void load()
        }}
      />

      <ImportTestsetModal
        open={importOpen}
        onClose={() => setImportOpen(false)}
        onCreated={() => {
          setImportOpen(false)
          void load()
        }}
      />

      <LogModal
        title={t('testset.logTitle')}
        open={logTestset !== null}
        onClose={() => setLogTestset(null)}
        entries={logData?.entries ?? []}
        error={logData?.error}
        textOf={(entry) => logEntryText(t, entry)}
      />
    </main>
  )
}
