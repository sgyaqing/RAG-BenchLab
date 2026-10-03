import { useEffect, useRef, useState } from 'react'
import {
  App as AntdApp,
  Button,
  Checkbox,
  Collapse,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Space,
  Tooltip,
} from 'antd'
import { CheckCircleFilled, CloseCircleFilled, QuestionCircleOutlined } from '@ant-design/icons'
import { useI18n } from '@/i18n'
import { listCorpora, detectCorpusLanguage, type Corpus } from '@/api/corpus'
import { listModelConfigs, type ModelConfig } from '@/api/modelConfig'
import { checkTestsetName, createTestset } from '@/api/testset'
import styles from './CreateCorpusModal.module.css'

const NAME_PATTERN = /^[\w\- 一-鿿]+$/u

interface Props {
  open: boolean
  onClose: () => void
  onCreated: () => void
}

interface FormValues {
  name: string
  corpus_id: number
  reuse_kg: boolean
  llm_config_id: number
  llm_concurrency: number
  llm_max_tokens: number
  embedding_config_id: number
  n_single: number
  n_multi_specific: number
  n_multi_abstract: number
  prompt_language: 'auto' | 'zh' | 'en'
  amplify: number
  gen_amplify: number
}

type NameCheck = { status: '' | 'validating' | 'success' | 'error'; help?: string }

export default function CreateTestsetModal({ open, onClose, onCreated }: Props) {
  const { t, locale } = useI18n()
  const { notification, modal } = AntdApp.useApp()
  const [form] = Form.useForm<FormValues>()

  const [corpora, setCorpora] = useState<Corpus[]>([])
  const [llms, setLlms] = useState<ModelConfig[]>([])
  const [embeddings, setEmbeddings] = useState<ModelConfig[]>([])
  const [nameCheck, setNameCheck] = useState<NameCheck>({ status: '' })
  const [creating, setCreating] = useState(false)
  const debounceTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  const corpusId = Form.useWatch('corpus_id', form)
  const reuseKg = Form.useWatch('reuse_kg', form)
  const selectedCorpus = corpora.find((c) => c.id === corpusId)
  const [detectedLang, setDetectedLang] = useState<'zh' | 'en' | null>(null)

  // Detect the corpus language as soon as a corpus is chosen (fast, no LLM).
  useEffect(() => {
    setDetectedLang(null)
    if (!corpusId) return
    void detectCorpusLanguage(corpusId).then((res) => setDetectedLang(res.language))
  }, [corpusId])

  const langLabel = (lang: 'zh' | 'en') =>
    lang === 'zh' ? t('testset.logLangZh') : t('testset.logLangEn')

  useEffect(() => {
    if (!open) return
    reset()
    void listCorpora('', 1, 100).then((res) =>
      setCorpora(res.items.filter((c) => c.status === 'completed' && c.success_files > 0)),
    )
    void listModelConfigs('', 1, 100).then((res) => {
      setLlms(res.items.filter((m) => m.type === 'llm'))
      setEmbeddings(res.items.filter((m) => m.type === 'embedding'))
    })
  }, [open])

  function reset() {
    form.resetFields()
    setNameCheck({ status: '' })
  }

  function onNameChange(value: string) {
    setNameCheck({ status: '' })
    clearTimeout(debounceTimer.current)
    const name = value.trim()
    if (!name || !NAME_PATTERN.test(name)) return
    debounceTimer.current = setTimeout(async () => {
      setNameCheck({ status: 'validating' })
      try {
        const res = await checkTestsetName(name)
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

  async function doCreate(values: FormValues) {
    setCreating(true)
    try {
      await createTestset(values)
      onCreated()
    } catch (e) {
      notification.error({
        message: t('testset.createFailed'),
        description: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
    } finally {
      setCreating(false)
    }
  }

  /** Guards every path that creates, taken before the name round trip: that
   * trip is long enough for a second click to start a second creation. */
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
        check = await checkTestsetName(name)
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
      // Pre-estimate: warn when the corpus cannot support N quality QA pairs.
      // (Prompt language is an explicit form field, so no extra confirmation.)
      const corpus = corpora.find((c) => c.id === values.corpus_id)
      const n = values.n_single + values.n_multi_specific + values.n_multi_abstract
      const maxQuality = Math.floor((corpus?.success_files ?? 0) / 1.3)
      const tooMany = corpus !== undefined && n > maxQuality

      if (tooMany) {
        modal.confirm({
          title: t('testset.createTitle'),
          content: t('testset.estimateWarning', { max: maxQuality }),
          okText: t('testset.estimateContinue'),
          cancelText: t('testset.estimateAdjust'),
          // antd holds this button in loading while the promise is pending, so
          // releasing the guard as this handler returns is safe here.
          onOk: () => doCreate(values),
        })
        return
      }
      await doCreate(values)
    } finally {
      submitting.current = false
      // Every path out of this function has to release the button. It was set
      // here and cleared only inside doCreate, so the four early returns above
      // — a form that fails validation, a name already taken, a name check
      // that errors, the too-many-pairs confirm — left it spinning for good,
      // and reopening the dialog did not clear it either: the reset on open
      // touches the form and the name check, not this.
      setCreating(false)
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
      title={t('testset.createTitle')}
      open={open}
      maskClosable={false}
      onCancel={onClose}
      footer={[
        <Button key="cancel" onClick={onClose}>
          {t('testset.cancel')}
        </Button>,
        <Button key="create" type="primary" loading={creating} onClick={() => void onCreate()}>
          {t('testset.createAction')}
        </Button>,
      ]}
      width={560}
    >
      {/* English labels are longer than the Chinese ones and the column clips
          rather than wraps: "Multi-hop Abstract Questions" needs 208px with
          its required asterisk, so 130px cut three of them off. 220px leaves
          the control column 292px, which the numbers and selects still fit. */}
      <Form
        form={form}
        className={styles.form}
        layout="horizontal"
        labelCol={{ flex: locale === 'zh' ? '130px' : '220px' }}
        wrapperCol={{ flex: '1 1 0%' }}
        colon={false}
        initialValues={{
          reuse_kg: true,
          llm_concurrency: 16,
          llm_max_tokens: 16384,
          n_single: 10,
          n_multi_specific: 10,
          n_multi_abstract: 10,
          prompt_language: 'auto',
          amplify: 1.3,
          gen_amplify: 1.6,
        }}
      >
        <Form.Item
          name="name"
          label={t('testset.nameLabel')}
          validateStatus={nameCheck.status}
          help={nameHelp}
          rules={[
            { required: true, message: t('corpus.nameRequired') },
            { pattern: NAME_PATTERN, message: t('corpus.nameInvalid') },
          ]}
        >
          <Input maxLength={64} onChange={(e) => onNameChange(e.target.value)} />
        </Form.Item>

        <Form.Item
          name="corpus_id"
          label={t('testset.corpusLabel')}
          rules={[{ required: true, message: t('testset.corpusRequired') }]}
        >
          <Select
            options={corpora.map((c) => ({ value: c.id, label: c.name }))}
            showSearch
            optionFilterProp="label"
          />
        </Form.Item>

        {/* No marginBottom on this row, on purpose. antd keeps a 24px bottom
            margin on every item and drops it to 0 when a help line appears,
            which is exactly the room the note takes — so the fields below stay
            put, as they do under the judge cache note. An inline margin beats
            that rule, and the note then shoves everything down by 14px. */}
        {selectedCorpus?.has_kg && (
          <Form.Item
            label=" "
            style={{ marginTop: -16 }}
            // Through antd's help slot, exactly as the judge cache note is, so
            // both notes sit on a line of their own under their checkbox.
            help={
              !reuseKg ? (
                <span className={styles.formNote}>{t('testset.rebuildKgNote')}</span>
              ) : undefined
            }
          >
            {/* The binding stays on the Checkbox and the icon sits outside it:
                anything among a Checkbox's own children toggles the box when
                clicked, so an icon in there would flip reuse_kg on its way to
                showing the tooltip. */}
            <Space size={6}>
              <Form.Item name="reuse_kg" valuePropName="checked" noStyle>
                <Checkbox>{t('testset.reuseKg')}</Checkbox>
              </Form.Item>
              <Tooltip title={t('testset.reuseKgTip')}>
                <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
              </Tooltip>
            </Space>
          </Form.Item>
        )}

        <Form.Item
          name="prompt_language"
          label={t('testset.promptLangLabel')}
          rules={[{ required: true }]}
        >
          <Select
            options={[
              {
                value: 'auto',
                label:
                  t('testset.promptLangAuto') +
                  (detectedLang
                    ? `（${t('testset.promptLangDetected', { lang: langLabel(detectedLang) })}）`
                    : ''),
              },
              { value: 'zh', label: langLabel('zh') },
              { value: 'en', label: langLabel('en') },
            ]}
          />
        </Form.Item>


        <Form.Item
          name="llm_config_id"
          label="LLM"
          rules={[{ required: true, message: t('testset.llmRequired') }]}
        >
          <Select
            options={llms.map((m) => ({ value: m.id, label: m.name }))}
            showSearch
            optionFilterProp="label"
          />
        </Form.Item>

        <Form.Item
          name="llm_max_tokens"
          label={t('testset.maxTokensLabel')}
          rules={[
            { required: true, type: 'integer', min: 4096, max: 32768, message: t('testset.maxTokensTooSmall') },
          ]}
        >
          <InputNumber min={4096} max={32768} style={{ width: '100%' }} />
        </Form.Item>

        <Form.Item
          name="embedding_config_id"
          label="Embedding"
          rules={[{ required: true, message: t('testset.embeddingRequired') }]}
        >
          <Select
            options={embeddings.map((m) => ({ value: m.id, label: m.name }))}
            showSearch
            optionFilterProp="label"
          />
        </Form.Item>

        {/* Below Embedding because the number governs both: it is the ceiling
            on LLM and Embedding calls alike, so it belongs to neither alone. */}
        <Form.Item
          name="llm_concurrency"
          label={
            <span>
              {t('testset.concurrencyLabel')}{' '}
              <Tooltip title={t('testset.concurrencyTip')}>
                <QuestionCircleOutlined style={{ color: '#98a2b3' }} />
              </Tooltip>
            </span>
          }
          rules={[{ required: true, type: 'integer', min: 1, max: 64 }]}
        >
          <InputNumber min={1} max={64} style={{ width: '100%' }} />
        </Form.Item>

        <Form.Item
          name="n_single"
          label={t('testset.nSingleLabel')}
          rules={[{ required: true, type: 'integer', min: 0, max: 1000 }]}
        >
          <InputNumber min={0} max={1000} style={{ width: '100%' }} />
        </Form.Item>

        <Form.Item
          name="n_multi_specific"
          label={t('testset.nMultiSpecificLabel')}
          rules={[{ required: true, type: 'integer', min: 0, max: 1000 }]}
        >
          <InputNumber min={0} max={1000} style={{ width: '100%' }} />
        </Form.Item>

        <Form.Item
          name="n_multi_abstract"
          label={t('testset.nMultiAbstractLabel')}
          rules={[{ required: true, type: 'integer', min: 0, max: 1000 }]}
        >
          <InputNumber min={0} max={1000} style={{ width: '100%' }} />
        </Form.Item>

        <Collapse
          ghost
          items={[
            {
              key: 'advanced',
              label: t('testset.advancedOptions'),
              children: (
                <>
                  <Form.Item
                    name="amplify"
                    label={t('testset.amplifyLabel')}
                    tooltip={t('testset.amplifyTooltip')}
                    rules={[{ required: true, type: 'number', min: 1.0, max: 3.0 }]}
                  >
                    <InputNumber min={1.0} max={3.0} step={0.1} style={{ width: '100%' }} />
                  </Form.Item>
                  <Form.Item
                    name="gen_amplify"
                    label={t('testset.genAmplifyLabel')}
                    tooltip={t('testset.genAmplifyTooltip')}
                    rules={[{ required: true, type: 'number', min: 1.0, max: 2.0 }]}
                  >
                    <InputNumber min={1.0} max={2.0} step={0.1} style={{ width: '100%' }} />
                  </Form.Item>
                </>
              ),
            },
          ]}
        />
      </Form>
    </Modal>
  )
}
