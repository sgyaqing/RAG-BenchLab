import { useCallback, useEffect, useRef, useState } from 'react'
import {
  App as AntdApp,
  Button,
  Form,
  Input,
  Modal,
  Popconfirm,
  Progress,
  Radio,
  Select,
  Table,
  Tag,
  Tooltip,
} from 'antd'
import {
  CheckCircleFilled,
  CloseCircleFilled,
  DeleteOutlined,
  EditOutlined,
  ExclamationCircleOutlined,
  FileTextOutlined,
  PlusOutlined,
  QuestionCircleOutlined,
  SearchOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useI18n, type TFunc, type TParams } from '@/i18n'
import { listModelConfigs, type ModelConfig } from '@/api/modelConfig'
import {
  checkRagSystemName,
  createRagSystem,
  deleteRagSystem,
  getRagSystemLog,
  listRagSystems,
  updateRagSystem,
  type RagSystem,
  type RagSystemLog,
  type RagSystemLogEntry,
} from '@/api/ragSystem'
import styles from './RagSystemPage.module.css'
import LogModal from './LogModal'
import listStyles from './CorpusPage.module.css'

const POLL_INTERVAL_MS = 1500
const NAME_CHECK_DELAY_MS = 300
const NAME_PATTERN = /^[一-鿿\w\- ]+$/

interface FormValues {
  name: string
  mode: 'smart' | 'manual'
  base_url: string
  api_key?: string
  headers?: string
  body_template: string
  answer_path: string
  contexts_path?: string
  llm_config_id?: number
  platform_hint?: string
}

type NameCheck = { status: 'success' | 'error' | 'validating' | ''; help?: string }

function formatDateTime(iso: string | null): string {
  if (!iso) return ''
  // Backend stores UTC without a tz marker; parse as UTC, display local.
  const d = new Date(/[zZ]|[+-]\d{2}:?\d{2}$/.test(iso) ? iso : iso + 'Z')
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

/** Map backend stage codes to i18n keys: testing -> stageTesting. */
export function ragStageText(t: TFunc, rs: RagSystem): string {
  const key = `ragSystem.stage${rs.stage[0].toUpperCase()}${rs.stage.slice(1)}`
  const text = t(key)
  // A stale cached bundle can lack a newly added stage: never show the
  // internal code to the user.
  return text === key ? t('common.processing') : text
}

/** One log line, or null when the locale table has no copy for this key — a
 * key the running bundle predates, which should not be shown as raw text.
 * A failure carries a `code`; the specific explanation wins over the generic
 * one, and the addresses that never worked are appended as a hint. */
export function ragLogEntryText(t: TFunc, entry: RagSystemLogEntry): string | null {
  const key = `ragSystem.log${entry.key[0].toUpperCase()}${entry.key.slice(1)}`
  const params = Object.fromEntries(
    Object.entries(entry.params).filter(([, v]) => v != null),
  ) as TParams
  if (entry.key === 'failed') {
    let text: string | undefined
    if (typeof params.code === 'string' && params.code) {
      const codeKey = `ragSystem.logFailed_${params.code}`
      const codeText = t(codeKey, params)
      if (codeText !== codeKey) text = codeText
    }
    text = text ?? t(key, params)
    if (typeof params.urls === 'string' && params.urls) {
      text += t('ragSystem.logFailedUrls', { urls: params.urls })
    }
    return text
  }
  const text = t(key, params)
  return text === key ? null : text
}

export default function RagSystemPage() {
  const { t, locale } = useI18n()
  const { notification } = AntdApp.useApp()
  const [form] = Form.useForm<FormValues>()

  const [data, setData] = useState<RagSystem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [searchName, setSearchName] = useState('')
  const [loading, setLoading] = useState(false)
  const [modalOpen, setModalOpen] = useState(false)
  const [editing, setEditing] = useState<RagSystem | null>(null)
  const [saving, setSaving] = useState(false)
  const [llmOptions, setLlmOptions] = useState<ModelConfig[]>([])
  const [nameCheck, setNameCheck] = useState<NameCheck>({ status: '' })
  const nameTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const [logTarget, setLogTarget] = useState<RagSystem | null>(null)
  const [logData, setLogData] = useState<RagSystemLog | null>(null)

  const mode = Form.useWatch('mode', form) ?? 'smart'

  // Typing in the search box fires a request per keystroke; without this a slow
  // earlier response would land last and show results for a stale keyword.
  const loadSeq = useRef(0)

  const load = useCallback(async () => {
    const seq = ++loadSeq.current
    setLoading(true)
    try {
      const res = await listRagSystems(searchName, page, pageSize)
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

  const hasConfiguring = data.some((r) => r.status === 'configuring')
  useEffect(() => {
    if (!hasConfiguring) return
    const timer = setInterval(() => void load(), POLL_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [hasConfiguring, load])

  function onNameChange(value: string) {
    setNameCheck({ status: '' })
    clearTimeout(nameTimer.current)
    const name = value.trim()
    // An empty or ill-formed name is a form-rule failure, not a taken one —
    // asking the server about it would only replace the rule's own message.
    if (!name || !NAME_PATTERN.test(name)) return
    nameTimer.current = setTimeout(async () => {
      setNameCheck({ status: 'validating' })
      try {
        const res = await checkRagSystemName(name, editing?.id)
        setNameCheck(
          res.available
            ? { status: 'success', help: t('corpus.nameAvailable', { name }) }
            : { status: 'error', help: t('corpus.nameUnavailable', { name }) },
        )
      } catch {
        setNameCheck({ status: 'error', help: t('corpus.nameCheckFailed') })
      }
    }, NAME_CHECK_DELAY_MS)
  }

  function openCreate() {
    setEditing(null)
    form.resetFields()
    form.setFieldsValue({ mode: 'smart' } as FormValues)
    setNameCheck({ status: '' })
    // The assist LLM list is fetched per open: a config added on the model page
    // since the last dialog would otherwise not be offered here.
    void listModelConfigs('', 1, 100).then((res) =>
      setLlmOptions(res.items.filter((c) => c.type === 'llm')),
    )
    setModalOpen(true)
  }

  function openEdit(record: RagSystem) {
    setEditing(record)
    form.resetFields()
    form.setFieldsValue({
      name: record.name,
      mode: 'manual',
      base_url: record.base_url,
      api_key: record.api_key ?? undefined,
      // "{}" is the stored form of "no extra headers"; showing it would make an
      // empty template look like a value the user has to keep.
      headers: record.headers === '{}' ? undefined : record.headers,
      body_template: record.body_template,
      answer_path: record.answer_path,
      contexts_path: record.contexts_path ?? undefined,
    })
    setNameCheck({ status: '' })
    setModalOpen(true)
  }

  /** Taken before the name round trip: that trip is long enough for a second
   * click to start a second adapter job. */
  const submitting = useRef(false)

  async function onSave() {
    if (submitting.current) return
    submitting.current = true
    setSaving(true)
    try {
      let values: FormValues
      try {
        values = await form.validateFields()
      } catch {
        return
      }
      const name = values.name.trim()
      let check: { available: boolean }
      try {
        check = await checkRagSystemName(name, editing?.id)
      } catch {
        // A rejection here used to escape into `void onSave()` and leave the
        // dialog sitting there with nothing to explain it.
        setNameCheck({ status: 'error', help: t('corpus.nameCheckFailed') })
        return
      }
      if (!check.available) {
        setNameCheck({ status: 'error', help: t('corpus.nameUnavailable', { name }) })
        return
      }
      try {
        if (editing) {
          await updateRagSystem(editing.id, {
            name,
            base_url: values.base_url,
            api_key: values.api_key || null,
            lang: locale,
            headers: values.headers || '{}',
            body_template: values.body_template,
            answer_path: values.answer_path,
            contexts_path: values.contexts_path?.trim() || null,
          })
        } else if (values.mode === 'smart') {
          await createRagSystem({
            mode: 'smart',
            name,
            base_url: values.base_url,
            api_key: values.api_key || null,
            lang: locale,
            llm_config_id: values.llm_config_id,
            platform_hint: values.platform_hint!.trim(),
          })
        } else {
          await createRagSystem({
            mode: 'manual',
            name,
            base_url: values.base_url,
            api_key: values.api_key || null,
            lang: locale,
            headers: values.headers || '{}',
            body_template: values.body_template,
            answer_path: values.answer_path,
            contexts_path: values.contexts_path?.trim() || null,
          })
        }
        // No "saved" toast: every one of these three paths then runs — smart
        // fill researches, manual fill probes — and the row it lands in shows
        // that with a progress bar. The corpus, testset and evaluation dialogs
        // say nothing on success for the same reason.
        setModalOpen(false)
        void load()
      } catch (e) {
        notification.error({
          message: (e as Error).message,
          placement: 'topRight',
          duration: 3,
        })
      }
    } finally {
      setSaving(false)
      submitting.current = false
    }
  }

  async function onDelete(record: RagSystem) {
    try {
      await deleteRagSystem(record.id)
      notification.success({
        message: t('ragSystem.deleteSuccess'),
        placement: 'topRight',
        duration: 3,
      })
      void load()
    } catch (e) {
      // a 409 (job still running) or 404 (removed elsewhere) used to be silent
      notification.error({
        message: t('common.deleteFailed'),
        description: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
    }
  }

  async function openLog(record: RagSystem) {
    setLogTarget(record)
    setLogData(null)
    try {
      setLogData(await getRagSystemLog(record.id))
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

  const nameHelp = nameCheck.help ? (
    <span className={styles.nameHelp}>
      {nameCheck.status === 'success' && (
        <CheckCircleFilled style={{ color: '#52c41a' }} />
      )}
      {nameCheck.status === 'error' && <CloseCircleFilled style={{ color: '#ff4d4f' }} />}
      <span className={styles.nameHelpText}>{nameCheck.help}</span>
    </span>
  ) : undefined

  const columns: ColumnsType<RagSystem> = [
    {
      title: t('ragSystem.name'),
      dataIndex: 'name',
      key: 'name',
      // A finished adapter with no contexts path is not broken — plenty of
      // systems never return the retrieved chunks. But it is the one loss the
      // customer can undo by hand, and the evaluation only mentions it after
      // they have already chosen to run. A muted mark, not an error colour:
      // this is a state, not a fault. Failed adapters never got that far, so
      // they carry nothing.
      render: (value: string, record: RagSystem) => (
        <>
          {value}
          {record.status === 'completed' && !record.contexts_path && (
            <Tooltip title={t('ragSystem.noContextsTip')}>
              <ExclamationCircleOutlined className={listStyles.nameMark} />
            </Tooltip>
          )}
        </>
      ),
    },
    {
      title: t('ragSystem.colStatus'),
      key: 'status',
      width: 260,
      render: (_value, record) =>
        record.status === 'completed' ? (
          <Tag color="success">{t('ragSystem.statusCompleted')}</Tag>
        ) : record.status === 'failed' ? (
          <Tag color="error">{t('ragSystem.statusFailed')}</Tag>
        ) : (
          <span className={listStyles.progressWrap}>
            <span className={listStyles.progressCell}>
              <Progress
                percent={Math.round(record.progress)}
                size="small"
                showInfo={false}
                status="active"
                strokeColor="#1677ff"
              />
              <span className={listStyles.progressText}>{Math.round(record.progress)}%</span>
            </span>
            <span className={listStyles.stageText}>{ragStageText(t, record)}</span>
          </span>
        ),
    },
    {
      title: t('ragSystem.colCompletedAt'),
      key: 'completedAt',
      render: (_value, record) => formatDateTime(record.completed_at),
    },
    {
      title: t('ragSystem.colActions'),
      key: 'actions',
      render: (_value, record) =>
        record.status === 'configuring' ? null : (
          <span>
            <Button
              type="link"
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => void openLog(record)}
            >
              {t('ragSystem.log')}
            </Button>
            <Button
              type="link"
              size="small"
              icon={<EditOutlined />}
              onClick={() => openEdit(record)}
            >
              {t('ragSystem.edit')}
            </Button>
            <Popconfirm
              title={t('ragSystem.deleteConfirm')}
              onConfirm={() => void onDelete(record)}
            >
              <Button type="link" size="small" danger icon={<DeleteOutlined />}>
                {t('ragSystem.delete')}
              </Button>
            </Popconfirm>
          </span>
        ),
    },
  ]

  // The assist-LLM fields only make sense for smart fill on a new adapter:
  // editing always goes through the manual fields.
  const showSmartFields = mode === 'smart' && !editing

  return (
    <main className={styles.page}>
      <div className={styles.toolbar}>
        <Input
          allowClear
          prefix={<SearchOutlined />}
          placeholder={t('ragSystem.searchPlaceholder')}
          style={{ width: 280 }}
          onChange={(e) => {
            setPage(1)
            setSearchName(e.target.value)
          }}
        />
        <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
          {t('ragSystem.create')}
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

      <Modal
        title={t(editing ? 'ragSystem.editTitle' : 'ragSystem.createTitle')}
        open={modalOpen}
        maskClosable={false}
        onCancel={() => setModalOpen(false)}
        width={640}
        footer={[
          <Button key="cancel" onClick={() => setModalOpen(false)}>
            {t('ragSystem.cancel')}
          </Button>,
          <Button key="save" type="primary" loading={saving} onClick={() => void onSave()}>
            {t(editing ? 'ragSystem.confirm' : 'ragSystem.createAction')}
          </Button>,
        ]}
      >
        <Form
          form={form}
          layout="horizontal"
          labelCol={{ flex: locale === 'zh' ? '130px' : '200px' }}
          wrapperCol={{ flex: '1 1 0%' }}
          colon={false}
        >
          <Form.Item
            name="name"
            label={t('ragSystem.name')}
            validateStatus={nameCheck.status}
            help={nameHelp}
            rules={[{ required: true, message: t('ragSystem.nameRequired') }]}
          >
            <Input onChange={(e) => onNameChange(e.target.value)} />
          </Form.Item>

          {!editing && (
            <Form.Item name="mode" label={t('ragSystem.configMethod')} rules={[{ required: true }]}>
              <Radio.Group>
                <Radio value="smart">{t('ragSystem.smart')}</Radio>
                <Radio value="manual">{t('ragSystem.manual')}</Radio>
              </Radio.Group>
            </Form.Item>
          )}

          {showSmartFields && (
            <>
              <Form.Item
                name="llm_config_id"
                label={t('ragSystem.assistLlm')}
                rules={[{ required: true, message: t('ragSystem.assistLlmRequired') }]}
              >
                <Select options={llmOptions.map((c) => ({ value: c.id, label: c.name }))} />
              </Form.Item>

              <Form.Item
                name="platform_hint"
                label={
                  <span>
                    {t('ragSystem.platformHint')}{' '}
                    <Tooltip title={t('ragSystem.platformHintTooltip')}>
                      <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
                    </Tooltip>
                  </span>
                }
                rules={[{ required: true, message: t('ragSystem.platformHintRequired') }]}
              >
                <Input placeholder={t('ragSystem.platformHintPlaceholder')} />
              </Form.Item>

              <Form.Item
                name="base_url"
                label={
                  <span>
                    {t('ragSystem.baseUrl')}{' '}
                    <Tooltip title={t('ragSystem.baseUrlTooltipSmart')}>
                      <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
                    </Tooltip>
                  </span>
                }
                rules={[{ required: true, message: t('ragSystem.baseUrlRequired') }]}
              >
                <Input placeholder={t('ragSystem.baseUrlPlaceholder')} autoComplete="off" />
              </Form.Item>

              <Form.Item
                name="api_key"
                label={
                  <span>
                    {t('ragSystem.apiKey')}{' '}
                    <Tooltip title={t('ragSystem.apiKeyTip')}>
                      <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
                    </Tooltip>
                  </span>
                }
              >
                <Input.Password autoComplete="new-password" />
              </Form.Item>
            </>
          )}

          {!showSmartFields && (
            <>
              <Form.Item
                name="headers"
                label={
                  <span>
                    {t('ragSystem.headers')}{' '}
                    <Tooltip title={t('ragSystem.headersHint')}>
                      <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
                    </Tooltip>
                  </span>
                }
                rules={[
                  {
                    validator: (_rule, value: string) => {
                      if (!value?.trim()) return Promise.resolve()
                      try {
                        const parsed = JSON.parse(value)
                        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
                          return Promise.resolve()
                        }
                      } catch {
                        // fall through to the rejection below
                      }
                      return Promise.reject(new Error(t('ragSystem.headersInvalid')))
                    },
                  },
                ]}
              >
                <Input.TextArea
                  rows={2}
                  style={{ fontFamily: 'monospace' }}
                  placeholder={'{"Authorization": "Bearer {{api_key}}"}'}
                />
              </Form.Item>

              <Form.Item
                name="body_template"
                label={
                  <span>
                    {t('ragSystem.bodyTemplate')}{' '}
                    <Tooltip title={t('ragSystem.bodyTemplateHint')}>
                      <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
                    </Tooltip>
                  </span>
                }
                rules={[{ required: true, message: t('ragSystem.bodyTemplateRequired') }]}
              >
                <Input.TextArea
                  rows={4}
                  style={{ fontFamily: 'monospace' }}
                  placeholder={'{"question": "{{question}}", "stream": false}'}
                />
              </Form.Item>

              <Form.Item
                name="answer_path"
                label={
                  <span>
                    {t('ragSystem.answerPath')}{' '}
                    <Tooltip title={t('ragSystem.pathHint')}>
                      <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
                    </Tooltip>
                  </span>
                }
                rules={[{ required: true, message: t('ragSystem.answerPathRequired') }]}
              >
                <Input placeholder="data.answer" />
              </Form.Item>

              <Form.Item
                name="contexts_path"
                label={
                  <span>
                    {t('ragSystem.contextsPath')}{' '}
                    <Tooltip title={t('ragSystem.contextsOptional')}>
                      <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
                    </Tooltip>
                  </span>
                }
              >
                <Input placeholder="data.reference.chunks[].content" />
              </Form.Item>

              <Form.Item
                name="base_url"
                label={
                  <span>
                    {t('ragSystem.baseUrl')}{' '}
                    <Tooltip
                      title={t(
                        editing
                          ? 'ragSystem.baseUrlTooltipEdit'
                          : 'ragSystem.baseUrlTooltipManual',
                      )}
                    >
                      <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
                    </Tooltip>
                  </span>
                }
                rules={[{ required: true, message: t('ragSystem.baseUrlRequired') }]}
              >
                <Input placeholder={t('ragSystem.baseUrlPlaceholder')} autoComplete="off" />
              </Form.Item>

              <Form.Item
                name="api_key"
                label={
                  <span>
                    {t('ragSystem.apiKey')}{' '}
                    <Tooltip title={t('ragSystem.apiKeyTip')}>
                      <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
                    </Tooltip>
                  </span>
                }
              >
                <Input.Password autoComplete="new-password" />
              </Form.Item>
            </>
          )}
        </Form>
      </Modal>

      <LogModal
        title={t('ragSystem.logTitle')}
        open={logTarget !== null}
        onClose={() => setLogTarget(null)}
        entries={logData?.entries ?? []}
        error={logData?.error}
        textOf={(entry) => ragLogEntryText(t, entry)}
      />
    </main>
  )
}
