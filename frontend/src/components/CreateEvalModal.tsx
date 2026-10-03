import { useEffect, useRef, useState } from 'react'
import {
  App as AntdApp,
  Button,
  Checkbox,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Tooltip,
} from 'antd'
import { CheckCircleFilled, CloseCircleFilled, QuestionCircleOutlined } from '@ant-design/icons'
import { useI18n } from '@/i18n'
import {
  checkEvalName,
  createEvaluation,
  METRIC_KEYS,
  metricDisplayName,
} from '@/api/evaluation'
import { listModelConfigs, type ModelConfig } from '@/api/modelConfig'
import { listRagSystems, type RagSystem } from '@/api/ragSystem'
import { listTestsets, type Testset } from '@/api/testset'
import styles from './CreateEvalModal.module.css'

interface Props {
  open: boolean
  onClose: () => void
  onCreated: () => void
}

interface FormValues {
  name: string
  testset_id: number
  rag_system_id: number
  base_url: string
  api_key?: string
  llm_config_id: number
  embedding_config_id: number
  concurrency: number
  judge_concurrency: number
  timeout: number
  use_judge_cache: boolean
  metrics: string[]
}

type NameCheck = { status: '' | 'validating' | 'success' | 'error'; help?: string }

const NAME_PATTERN = /^[一-鿿\w\- ]+$/

const CONTEXT_METRICS = new Set(['faithfulness', 'context_precision', 'context_recall'])

export default function CreateEvalModal({ open, onClose, onCreated }: Props) {
  const { t, locale } = useI18n()
  const { notification } = AntdApp.useApp()
  const [form] = Form.useForm<FormValues>()

  const [testsets, setTestsets] = useState<Testset[]>([])
  const [ragSystems, setRagSystems] = useState<RagSystem[]>([])
  const [llms, setLlms] = useState<ModelConfig[]>([])
  const [embeddings, setEmbeddings] = useState<ModelConfig[]>([])
  const [nameCheck, setNameCheck] = useState<NameCheck>({ status: '' })
  const [creating, setCreating] = useState(false)
  const debounceTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  const ragSystemId = Form.useWatch('rag_system_id', form)
  const useJudgeCache = Form.useWatch('use_judge_cache', form)
  const selectedRag = ragSystems.find((r) => r.id === ragSystemId)
  const ragHasContexts = Boolean(selectedRag?.contexts_path)

  // Adapter without a contexts path: context metrics are unchecked AND
  // disabled (a disabled-but-checked box would still be submitted, and the
  // user must SEE they won't be evaluated). If they were auto-unchecked and
  // the user then picks a contexts-capable adapter, check them back — a run
  // with only Response Relevancy is rarely meaningful.
  const autoUnchecked = useRef(false)
  useEffect(() => {
    if (!selectedRag) return
    const current: string[] = form.getFieldValue('metrics') ?? []
    if (!ragHasContexts) {
      const next = current.filter((m) => !CONTEXT_METRICS.has(m))
      if (next.length !== current.length) {
        form.setFieldsValue({ metrics: next })
        autoUnchecked.current = true
      }
    } else if (autoUnchecked.current) {
      const merged = [...current]
      for (const m of METRIC_KEYS) {
        if (CONTEXT_METRICS.has(m) && !merged.includes(m)) merged.push(m)
      }
      form.setFieldsValue({ metrics: merged })
      autoUnchecked.current = false
    }
  }, [selectedRag, ragHasContexts, form])

  useEffect(() => {
    if (!open) return
    form.resetFields()
    autoUnchecked.current = false
    form.setFieldsValue({ concurrency: 4, judge_concurrency: 16, timeout: 120, metrics: [...METRIC_KEYS], use_judge_cache: true })
    setNameCheck({ status: '' })
    void listTestsets('', 1, 100).then((res) =>
      setTestsets(res.items.filter((ts) => ts.status === 'completed')),
    )
    void listRagSystems('', 1, 100).then((res) =>
      setRagSystems(res.items.filter((r) => r.status === 'completed')),
    )
    void listModelConfigs('', 1, 100).then((res) => {
      setLlms(res.items.filter((m) => m.type === 'llm'))
      setEmbeddings(res.items.filter((m) => m.type === 'embedding'))
    })
  }, [open])

  function onNameChange(value: string) {
    setNameCheck({ status: '' })
    clearTimeout(debounceTimer.current)
    const name = value.trim()
    if (!name || !NAME_PATTERN.test(name)) return
    debounceTimer.current = setTimeout(async () => {
      setNameCheck({ status: 'validating' })
      try {
        const res = await checkEvalName(name)
        setNameCheck(
          res.available
            ? { status: 'success', help: t('corpus.nameAvailable', { name }) }
            : { status: 'error', help: t('corpus.nameUnavailable', { name }) },
        )
      } catch {
        setNameCheck({ status: 'error', help: t('corpus.nameCheckFailed') })
      }
    }, 300)
  }

  /** Taken before the name round trip: that trip is long enough for a second
   * click to start a second run. */
  const submitting = useRef(false)

  async function onCreate() {
    if (submitting.current) return
    submitting.current = true
    setCreating(true)
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
        check = await checkEvalName(name)
      } catch {
        // A rejection here used to escape into `void onCreate()` and leave the
        // dialog sitting there with nothing to explain it.
        setNameCheck({ status: 'error', help: t('corpus.nameCheckFailed') })
        return
      }
      if (!check.available) {
        setNameCheck({ status: 'error', help: t('corpus.nameUnavailable', { name }) })
        return
      }
      try {
        await createEvaluation({ ...values, name, api_key: values.api_key || null })
        onCreated()
      } catch (e) {
        notification.error({
          message: t('evaluation.createFailed'),
          description: (e as Error).message,
          placement: 'topRight',
          duration: 3,
        })
      }
    } finally {
      setCreating(false)
      submitting.current = false
    }
  }

  const nameHelp = nameCheck.help ? (
    <span className={styles.nameHelp}>
      {nameCheck.status === 'success' && <CheckCircleFilled style={{ color: '#52c41a' }} />}
      {nameCheck.status === 'error' && <CloseCircleFilled style={{ color: '#ff4d4f' }} />}
      <span className={styles.nameHelpText}>{nameCheck.help}</span>
    </span>
  ) : undefined

  return (
    <Modal
      title={t('evaluation.createTitle')}
      open={open}
      maskClosable={false}
      onCancel={onClose}
      width={locale === 'zh' ? 560 : 620}
      footer={[
        <Button key="cancel" onClick={onClose}>
          {t('evaluation.cancel')}
        </Button>,
        <Button
          key="create"
          type="primary"
          loading={creating}
          onClick={() => void onCreate()}
        >
          {t('evaluation.createAction')}
        </Button>,
      ]}
    >
      {/* Two things change in English: the labels need a wider column
          ("RAG Call Concurrency" needs 176px with its asterisk, against 130px
          of room plus only 20px of slack in the row it shares), and the dialog
          has to grow to pay for it. Chinese keeps the original 560/130. */}
      <Form
        form={form}
        layout="horizontal"
        labelCol={{ flex: locale === 'zh' ? '130px' : '180px' }}
        wrapperCol={{ flex: '1 1 0%' }}
        colon={false}
      >
        <Form.Item
          name="name"
          label={t('evaluation.nameLabel')}
          validateStatus={nameCheck.status}
          help={nameHelp}
          rules={[{ required: true, message: t('evaluation.nameRequired') }]}
        >
          <Input onChange={(e) => onNameChange(e.target.value)} />
        </Form.Item>

        <Form.Item
          name="testset_id"
          label={t('evaluation.testsetLabel')}
          rules={[{ required: true, message: t('evaluation.testsetRequired') }]}
        >
          <Select
            options={testsets.map((ts) => ({ value: ts.id, label: ts.name }))}
            placeholder={t('evaluation.testsetRequired')}
          />
        </Form.Item>

        <Form.Item
          name="rag_system_id"
          label={t('evaluation.ragSystemLabel')}
          rules={[{ required: true, message: t('evaluation.ragSystemRequired') }]}
        >
          <Select
            options={ragSystems.map((r) => ({ value: r.id, label: r.name }))}
            placeholder={t('evaluation.ragSystemRequired')}
          />
        </Form.Item>

        <Form.Item
          name="base_url"
          label={
            <span>
              {t('evaluation.endpointUrl')}{' '}
              <Tooltip title={t('evaluation.endpointUrlTip')}>
                <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
              </Tooltip>
            </span>
          }
          rules={[{ required: true, message: t('evaluation.endpointUrlRequired') }]}
        >
          <Input autoComplete="off" />
        </Form.Item>

        <Form.Item
          name="api_key"
          label={
            <span>
              {t('evaluation.apiKeyOverride')}{' '}
              <Tooltip title={t('evaluation.apiKeyOverrideTip')}>
                <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
              </Tooltip>
            </span>
          }
        >
          <Input.Password autoComplete="new-password" />
        </Form.Item>

        <Form.Item
          name="llm_config_id"
          label={t('evaluation.judgeLlmLabel')}
          rules={[{ required: true, message: t('evaluation.judgeLlmRequired') }]}
        >
          <Select options={llms.map((m) => ({ value: m.id, label: m.name }))} />
        </Form.Item>

        <Form.Item
          name="embedding_config_id"
          label={t('evaluation.judgeEmbLabel')}
          rules={[{ required: true, message: t('evaluation.judgeEmbRequired') }]}
        >
          <Select options={embeddings.map((m) => ({ value: m.id, label: m.name }))} />
        </Form.Item>

        <Form.Item
          name="use_judge_cache"
          valuePropName="checked"
          label={
            <span>
              {t('evaluation.judgeCacheLabel')}{' '}
              <Tooltip title={t('evaluation.useJudgeCacheTip')}>
                <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
              </Tooltip>
            </span>
          }
          // Rendered through antd's help slot so the spacing matches every
          // other inline hint in this form.
          help={
            useJudgeCache === false ? (
              <span className={styles.formNote}>{t('evaluation.judgeCacheOffNote')}</span>
            ) : undefined
          }
        >
          <Checkbox>{t('evaluation.useJudgeCache')}</Checkbox>
        </Form.Item>

        <Form.Item
          name="metrics"
          label={t('evaluation.metricsLabel')}
          rules={[{ required: true, message: t('evaluation.metricsRequired') }]}
          // Through antd's help slot, like every other inline hint in this
          // form: help is laid out inside the control column, so the note sits
          // under the checkboxes in both locales. As a block of its own it
          // needed a left margin hardcoded to the Chinese label column (130px)
          // and so sat 50px left of everything else in English (180px).
          help={
            selectedRag !== undefined && !ragHasContexts ? (
              <span className={styles.formNote}>
                {t('evaluation.metricNeedsContexts')}
              </span>
            ) : undefined
          }
        >
          <Checkbox.Group
            style={{ display: 'flex', flexDirection: 'column', gap: 8 }}
            options={METRIC_KEYS.map((key) => {
              const needContexts = CONTEXT_METRICS.has(key)
              const disabled = needContexts && selectedRag !== undefined && !ragHasContexts
              const label = (
                <Tooltip title={t(`evaluation.metricTip_${key}`)}>
                  {metricDisplayName(key)}
                </Tooltip>
              )
              return { value: key, label, disabled }
            })}
          />
        </Form.Item>
        <div style={{ display: 'flex', gap: 12 }}>
          <Form.Item
            name="concurrency"
            labelCol={{ flex: locale === 'zh' ? '0 0 140px' : '0 0 180px' }}
            label={
              <span>
                {t('evaluation.ragConcurrency')}{' '}
                <Tooltip title={t('evaluation.targetConcurrencyTip')}>
                  <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
                </Tooltip>
              </span>
            }
            rules={[{ required: true }]}
          >
            <InputNumber min={1} max={32} style={{ width: 90 }} />
          </Form.Item>
          <Form.Item
            name="judge_concurrency"
            labelCol={{ flex: 'none' }}
            wrapperCol={{ flex: 'none' }}
            label={
              <span>
                {t('evaluation.judgeConcurrency')}{' '}
                <Tooltip title={t('evaluation.judgeConcurrencyTip')}>
                  <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
                </Tooltip>
              </span>
            }
            rules={[{ required: true }]}
          >
            <InputNumber min={1} max={64} style={{ width: 90 }} />
          </Form.Item>
        </div>

        <Form.Item
          name="timeout"
          label={
            <span>
              {t('evaluation.timeout')}{' '}
              <Tooltip title={t('evaluation.timeoutTip')}>
                <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
              </Tooltip>
            </span>
          }
          rules={[{ required: true }]}
        >
          <InputNumber min={5} max={600} style={{ width: 120 }} />
        </Form.Item>
      </Form>
    </Modal>
  )
}
