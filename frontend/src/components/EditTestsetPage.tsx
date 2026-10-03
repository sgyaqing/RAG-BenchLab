import { useCallback, useEffect, useRef, useState } from 'react'
import {
  App as AntdApp,
  Button,
  Input,
  Modal,
  Popconfirm,
  Table,
  Tag,
} from 'antd'
import { CopyOutlined, DeleteOutlined, ReadOutlined, SnippetsOutlined, ArrowLeftOutlined } from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useParams, useNavigate, useLocation } from 'react-router-dom'
import { useI18n } from '@/i18n'
import {
  deleteTestsetItem,
  firstContextPreview,
  getTestset,
  listTestsetItems,
  restoreTestsetItem,
  updateTestsetItem,
  type TestsetItem,
} from '@/api/testset'
import styles from './EditTestsetPage.module.css'

function typeLabel(t: (key: string) => string, synthesizerName: string): string {
  if (synthesizerName.includes('multi_hop_abstract')) return t('testset.typeShortMultiAbstract')
  if (synthesizerName.includes('multi_hop')) return t('testset.typeShortMultiSpecific')
  if (synthesizerName.includes('single_hop')) return t('testset.typeShortSingle')
  return '' // imported items may carry no/unknown type — leave blank
}

export default function EditTestsetPage() {
  const { t } = useI18n()
  const { notification } = AntdApp.useApp()
  const { id } = useParams<{ id: string }>()
  const testsetId = Number(id)
  const navigate = useNavigate()
  const location = useLocation()

  const [testsetName, setTestsetName] = useState('')
  const [data, setData] = useState<TestsetItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [loading, setLoading] = useState(false)

  const [detailItem, setDetailItem] = useState<TestsetItem | null>(null)
  const [draftQuestion, setDraftQuestion] = useState('')
  const [draftReference, setDraftReference] = useState('')
  const [saving, setSaving] = useState(false)

  // Paging faster than the responses return would otherwise let an earlier page
  // land last and replace the one the user is on.
  const loadSeq = useRef(0)

  const load = useCallback(async () => {
    const seq = ++loadSeq.current
    setLoading(true)
    try {
      const res = await listTestsetItems(testsetId, page, pageSize)
      if (seq !== loadSeq.current) return
      setData(res.items)
      setTotal(res.total)
    } finally {
      if (seq === loadSeq.current) setLoading(false)
    }
  }, [testsetId, page, pageSize])

  useEffect(() => {
    void getTestset(testsetId).then((ts) => setTestsetName(ts.name))
  }, [testsetId])

  useEffect(() => {
    void load()
  }, [load])

  function openDetail(item: TestsetItem) {
    setDetailItem(item)
    setDraftQuestion(item.user_input)
    setDraftReference(item.reference)
  }

  // The callers invoke these with `void onX()`, so a rejection used to
  // disappear: the backend's 422 (empty question) or 404 (deleted in another
  // tab) left the dialog sitting there with nothing to explain it.
  function reportFailure(message: string, e: unknown) {
    notification.error({
      message,
      description: (e as Error).message,
      placement: 'topRight',
      duration: 3,
    })
  }

  async function onSave() {
    if (!detailItem) return
    setSaving(true)
    try {
      await updateTestsetItem(testsetId, detailItem.id, draftQuestion, draftReference)
      notification.success({
        message: t('testset.saveSuccess'),
        placement: 'topRight',
        duration: 3,
      })
      setDetailItem(null)
      void load()
    } catch (e) {
      reportFailure(t('common.saveFailed'), e)
    } finally {
      setSaving(false)
    }
  }

  async function onRestore() {
    if (!detailItem) return
    try {
      const restored = await restoreTestsetItem(testsetId, detailItem.id)
      setDetailItem(null)
      notification.success({ message: t('testset.saveSuccess'), placement: 'topRight', duration: 3 })
      void load()
      return restored
    } catch (e) {
      reportFailure(t('common.saveFailed'), e)
      return undefined
    }
  }

  async function onDelete(item: TestsetItem) {
    try {
      await deleteTestsetItem(testsetId, item.id)
      notification.success({
        message: t('testset.deleteSuccess'),
        placement: 'topRight',
        duration: 3,
      })
      void load()
    } catch (e) {
      reportFailure(t('common.deleteFailed'), e)
    }
  }

  const columns: ColumnsType<TestsetItem> = [
    { title: t('testset.colSeq'), dataIndex: 'seq', key: 'seq', width: 64 },
    {
      title: 'user_input',
      dataIndex: 'user_input',
      key: 'user_input',
      render: (v: string, r) => (
        <span className={styles.contextCell} title={v}>
          <span className={styles.contextPreview}>{v}</span>
          {r.edited && (
            <Tag className={styles.editedTag} color="warning">
              {t('testset.editedTag')}
            </Tag>
          )}
        </span>
      ),
    },
    { title: 'reference', dataIndex: 'reference', key: 'reference', ellipsis: true },
    {
      title: 'reference_contexts',
      key: 'contexts',
      render: (_, r) => (
        <span className={styles.contextCell}>
          <span className={styles.contextPreview}>
            {firstContextPreview(r.reference_contexts, 60)}
          </span>
          <span className={styles.contextBadge} title={t('testset.contextsReadOnly')}>
            <SnippetsOutlined /> {r.reference_contexts.length}
          </span>
        </span>
      ),
    },
    {
      title: t('testset.colType'),
      key: 'type',
      width: 110,
      render: (_, r) => typeLabel(t, r.synthesizer_name),
    },
    {
      title: t('testset.colActions'),
      key: 'actions',
      width: 170,
      render: (_, r) => (
        <span>
          <Button
            type="link"
            size="small"
            icon={<ReadOutlined />}
            onClick={() => openDetail(r)}
          >
            {t('testset.detail')}
          </Button>
          <Popconfirm
            title={t('testset.deleteRowConfirm')}
            onConfirm={() => void onDelete(r)}
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>
              {t('testset.delete')}
            </Button>
          </Popconfirm>
        </span>
      ),
    },
  ]

  return (
    <main className={styles.editPage}>
      <div className={styles.pageHeader}>
        <Button
          type="text"
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate('/testsets', { state: location.state })}
        >
          {t('common.back')}
        </Button>
        <h2 className={styles.pageTitle}>{t('testset.editTitle', { name: testsetName })}</h2>
      </div>

      <Table<TestsetItem>
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
        title={t('testset.detailTitle')}
        open={detailItem !== null}
        maskClosable={false}
        onCancel={() => setDetailItem(null)}
        width={720}
        footer={[
          detailItem?.has_original && (
            <Button key="restore" onClick={() => void onRestore()} style={{ float: 'left' }}>
              {t('testset.restore')}
            </Button>
          ),
          <Button key="cancel" onClick={() => setDetailItem(null)}>
            {t('testset.cancel')}
          </Button>,
          <Button key="save" type="primary" loading={saving} onClick={() => void onSave()}>
            {t('testset.save')}
          </Button>,
        ]}
      >
        {detailItem && (
          <div>
            <p>{t('testset.question')}</p>
            <Input.TextArea
              rows={3}
              value={draftQuestion}
              onChange={(e) => setDraftQuestion(e.target.value)}
            />
            <p>{t('testset.referenceAnswer')}</p>
            <Input.TextArea
              rows={4}
              value={draftReference}
              onChange={(e) => setDraftReference(e.target.value)}
            />
            <p className={styles.softWarning}>{t('testset.editSoftWarning')}</p>
            <p>
              {t('testset.contextsReadOnly')}
            </p>
            {detailItem.reference_contexts.map((ctx, i) => (
              <div key={i} className={styles.contextCard}>
                <div className={styles.contextCardHeader}>
                  <span>
                    {t('testset.segment', {
                      index: i + 1,
                      total: detailItem.reference_contexts.length,
                    })}
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
            ))}
          </div>
        )}
      </Modal>
    </main>
  )
}
