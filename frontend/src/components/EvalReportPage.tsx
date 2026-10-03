import { useCallback, useEffect, useRef, useState } from 'react'
import { App as AntdApp, Button, Card, Modal, Table, Tag } from 'antd'
import { CopyOutlined, ReadOutlined, ArrowLeftOutlined } from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useParams, useNavigate, useLocation } from 'react-router-dom'
import { useI18n } from '@/i18n'
import {
  getEvaluation,
  listEvaluationItems,
  metricDisplayName,
  type EvalItem,
  type EvalRun,
} from '@/api/evaluation'
import styles from './EvalReportPage.module.css'

function parseJson<T>(raw: string, fallback: T): T {
  try {
    return JSON.parse(raw) as T
  } catch {
    return fallback
  }
}

export default function EvalReportPage() {
  const { t } = useI18n()
  const { notification } = AntdApp.useApp()
  const { id } = useParams<{ id: string }>()
  const runId = Number(id)
  const navigate = useNavigate()
  const location = useLocation()

  const [run, setRun] = useState<EvalRun | null>(null)
  const [data, setData] = useState<EvalItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [loading, setLoading] = useState(false)
  const [detailItem, setDetailItem] = useState<EvalItem | null>(null)

  useEffect(() => {
    void getEvaluation(runId).then(setRun)
  }, [runId])

  // Paging faster than the responses return would otherwise let an earlier page
  // land last and replace the one the user is on.
  const loadSeq = useRef(0)

  const load = useCallback(async () => {
    const seq = ++loadSeq.current
    setLoading(true)
    try {
      const res = await listEvaluationItems(runId, page, pageSize)
      if (seq !== loadSeq.current) return
      setData(res.items)
      setTotal(res.total)
    } finally {
      if (seq === loadSeq.current) setLoading(false)
    }
  }, [runId, page, pageSize])

  useEffect(() => {
    void load()
  }, [load])

  const metricKeys = run ? (parseJson(run.metrics, [] as string[])) : []
  const summary = run ? parseJson(run.summary, {} as Record<string, number>) : {}

  const columns: ColumnsType<EvalItem> = [
    { title: t('evaluation.colSeq'), dataIndex: 'seq', key: 'seq', width: 64 },
    {
      title: t('evaluation.colQuestion'),
      dataIndex: 'user_input',
      key: 'user_input',
      ellipsis: true,
    },
    {
      title: t('evaluation.colAnswer'),
      key: 'answer',
      ellipsis: true,
      render: (_, r) =>
        r.answer ?? (
          <span className={styles.failedText}>
            {t('evaluation.itemFailedShort')}
          </span>
        ),
    },
    ...metricKeys.map(
      (key): NonNullable<ColumnsType<EvalItem>[number]> => ({
        title: metricDisplayName(key),
        key,
        width: 150,
        render: (_, r) => {
          const scores = parseJson(r.scores, {} as Record<string, number>)
          return key in scores ? scores[key] : '-'
        },
      }),
    ),
    {
      title: t('evaluation.colActions'),
      key: 'actions',
      width: 100,
      render: (_, r) => (
        <Button
          type="link"
          size="small"
          icon={<ReadOutlined />}
          onClick={() => setDetailItem(r)}
        >
          {t('evaluation.detail')}
        </Button>
      ),
    },
  ]

  const detailContexts = detailItem ? parseJson(detailItem.contexts, [] as string[]) : []
  const detailScores = detailItem ? parseJson(detailItem.scores, {} as Record<string, number>) : {}

  return (
    <main className={styles.page}>
      <div className={styles.pageHeader}>
        <Button
          type="text"
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate('/evaluations', { state: location.state })}
        >
          {t('common.back')}
        </Button>
        <h2 className={styles.pageTitle}>
          {t('evaluation.reportTitle', { name: run?.name ?? '' })}
        </h2>
      </div>

      <div className={styles.summaryRow}>
        {metricKeys.map((key) => (
          <Card key={key} size="small" className={styles.summaryCard}>
            <div className={styles.metricName}>{metricDisplayName(key)}</div>
            <div className={styles.metricValue}>
              {key in summary ? summary[key] : t('evaluation.metricNotEvaluated')}
            </div>
          </Card>
        ))}
        <Card size="small" className={styles.summaryCard}>
          <div className={styles.metricName}>{t('evaluation.summaryItems')}</div>
          <div className={styles.metricValue}>
            {run?.total_items ?? '-'}
            {run !== null && run.failed_items > 0 && (
              <span className={styles.failedText}>
                {' '}
                ({t('evaluation.failedItemsTag', { count: run.failed_items })})
              </span>
            )}
          </div>
        </Card>
      </div>

      <Table<EvalItem>
        rowKey="id"
        loading={loading}
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

      <Modal
        title={t('evaluation.detailTitle')}
        open={detailItem !== null}
        maskClosable={false}
        onCancel={() => setDetailItem(null)}
        width={720}
        footer={
          <Button key="close" onClick={() => setDetailItem(null)}>
            {t('evaluation.cancel')}
          </Button>
        }
      >
        {detailItem && (
          <div>
            <div className={styles.scoreRow}>
              {metricKeys.map((key) => (
                <Tag key={key}>
                  {metricDisplayName(key)}: {key in detailScores ? detailScores[key] : '-'}
                </Tag>
              ))}
            </div>
            <p className={styles.fieldLabel}>{t('evaluation.colQuestion')}</p>
            <p className={styles.fieldText}>{detailItem.user_input}</p>
            <p className={styles.fieldLabel}>{t('evaluation.sysAnswer')}</p>
            {detailItem.answer !== null ? (
              <p className={styles.fieldText}>{detailItem.answer}</p>
            ) : (
              <p className={styles.failedText}>
                {t('evaluation.itemError', { error: detailItem.error ?? '' })}
              </p>
            )}
            <p className={styles.fieldLabel}>{t('evaluation.refAnswer')}</p>
            <p className={styles.fieldText}>{detailItem.reference}</p>
            <p className={styles.fieldLabel}>{t('evaluation.retrievedContexts')}</p>
            {detailContexts.length === 0 ? (
              <p className={styles.fieldText}>{t('evaluation.noContexts')}</p>
            ) : (
              detailContexts.map((ctx, i) => (
                <div key={i} className={styles.contextCard}>
                  <div className={styles.contextCardHeader}>
                    <span>
                      {t('testset.segment', { index: i + 1, total: detailContexts.length })}
                    </span>
                    <Button
                      type="link"
                      size="small"
                      icon={<CopyOutlined />}
                      onClick={() => {
                        void navigator.clipboard.writeText(ctx)
                        notification.success({
                          message: t('testset.copied'),
                          placement: 'topRight',
                          duration: 2,
                        })
                      }}
                    >
                      {t('common.copy')}
                    </Button>
                  </div>
                  <pre className={styles.contextCardBody}>{ctx}</pre>
                </div>
              ))
            )}
          </div>
        )}
      </Modal>
    </main>
  )
}
