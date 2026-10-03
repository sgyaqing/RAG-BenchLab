import { useCallback, useEffect, useRef, useState } from 'react'
import { App as AntdApp, Button, Input, Popconfirm, Progress, Table, Tag } from 'antd'
import {
  DeleteOutlined,
  DownloadOutlined,
  FileTextOutlined,
  PlusOutlined,
  SearchOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useI18n } from '@/i18n'
import { downloadWithSaveAs } from '@/utils/download'
import { formatDateTime } from '@/utils/format'
import { logEntryText } from '@/utils/log'
import { deleteCorpus, getCorpusLog, listCorpora, type Corpus, type CorpusLog } from '@/api/corpus'
import CreateCorpusModal from './CreateCorpusModal'
import LogModal from './LogModal'
import styles from './CorpusPage.module.css'

const POLL_INTERVAL_MS = 1500

export default function CorpusPage() {
  const { t } = useI18n()
  const { notification } = AntdApp.useApp()

  const [data, setData] = useState<Corpus[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [searchName, setSearchName] = useState('')
  const [loading, setLoading] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [logTarget, setLogTarget] = useState<Corpus | null>(null)
  const [logData, setLogData] = useState<CorpusLog | null>(null)

  // Typing in the search box fires a request per keystroke; without this a slow
  // earlier response would land last and show results for a stale keyword.
  const loadSeq = useRef(0)

  const load = useCallback(async () => {
    const seq = ++loadSeq.current
    setLoading(true)
    try {
      const res = await listCorpora(searchName, page, pageSize)
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

  const hasConverting = data.some((c) => c.status === 'converting')
  useEffect(() => {
    if (!hasConverting) return
    const timer = setInterval(() => void load(), POLL_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [hasConverting, load])

  async function onDelete(record: Corpus) {
    try {
      await deleteCorpus(record.id)
      notification.success({
        message: t('corpus.deleteSuccess'),
        placement: 'topRight',
        duration: 3,
      })
      void load()
    } catch (e) {
      // a 409 (a testset is still generating from it) or 404 (removed elsewhere)
      // used to be a silent no-op: the row stayed and nothing said why
      notification.error({
        message: t('common.deleteFailed'),
        description: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
    }
  }

  async function onExport(record: Corpus) {
    try {
      await downloadWithSaveAs(`/api/corpora/${record.id}/export`, `${record.name}.zip`, {
        'application/zip': ['.zip'],
      })
    } catch {
      notification.error({
        message: t('corpus.exportFailed'),
        placement: 'topRight',
        duration: 3,
      })
    }
  }

  async function openLog(record: Corpus) {
    setLogTarget(record)
    setLogData(null)
    try {
      setLogData(await getCorpusLog(record.id))
    } catch (e) {
      notification.error({
        message: t('common.loadFailed'),
        description: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
      setLogTarget(null)
    }
  }

  const columns: ColumnsType<Corpus> = [
    { title: t('corpus.colName'), dataIndex: 'name', key: 'name' },
    {
      title: t('corpus.colFileCount'),
      key: 'fileCount',
      render: (_value, record) => (record.status === 'completed' ? record.success_files : null),
    },
    {
      title: t('corpus.colStatus'),
      key: 'status',
      width: 260,
      render: (_value, record) =>
        record.status === 'completed' ? (
          <Tag color="success">{t('corpus.statusCompleted')}</Tag>
        ) : (
          <span className={styles.progressCell}>
            <Progress
              percent={
                record.total_files > 0
                  ? Math.round((record.processed_files / record.total_files) * 100)
                  : 0
              }
              size="small"
              showInfo={false}
            />
            <span className={styles.progressText}>
              {record.processed_files}/{record.total_files}
            </span>
          </span>
        ),
    },
    {
      title: t('corpus.colCompletedAt'),
      key: 'completedAt',
      render: (_value, record) => formatDateTime(record.completed_at),
    },
    {
      title: t('corpus.colActions'),
      key: 'actions',
      render: (_value, record) =>
        record.status === 'completed' ? (
          <span>
            <Button
              type="link"
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => void openLog(record)}
            >
              {t('corpus.log')}
            </Button>
            <Button
              type="link"
              size="small"
              icon={<DownloadOutlined />}
              onClick={() => void onExport(record)}
            >
              {t('corpus.export')}
            </Button>
            <Popconfirm title={t('corpus.deleteConfirm')} onConfirm={() => void onDelete(record)}>
              <Button type="link" size="small" danger icon={<DeleteOutlined />}>
                {t('corpus.delete')}
              </Button>
            </Popconfirm>
          </span>
        ) : null,
    },
  ]

  return (
    <main className={styles.page}>
      <div className={styles.toolbar}>
        <Input
          allowClear
          prefix={<SearchOutlined />}
          placeholder={t('corpus.searchPlaceholder')}
          style={{ width: 280 }}
          onChange={(e) => {
            setPage(1)
            setSearchName(e.target.value)
          }}
        />
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateOpen(true)}>
          {t('corpus.create')}
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

      <CreateCorpusModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={() => {
          setCreateOpen(false)
          void load()
        }}
      />

      <LogModal
        title={t('corpus.logTitle')}
        open={logTarget !== null}
        onClose={() => setLogTarget(null)}
        entries={logData?.entries ?? []}
        textOf={(entry) => logEntryText(t, 'corpus', entry)}
      />
    </main>
  )
}
