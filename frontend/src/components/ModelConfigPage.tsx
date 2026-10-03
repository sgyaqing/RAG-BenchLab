import { useCallback, useEffect, useRef, useState } from 'react'
import {
  App as AntdApp,
  AutoComplete,
  Button,
  Checkbox,
  Form,
  Input,
  Modal,
  Popconfirm,
  Radio,
  Space,
  Table,
} from 'antd'
import {
  ApiOutlined,
  CheckCircleFilled,
  CloseCircleFilled,
  DeleteOutlined,
  DownOutlined,
  EditOutlined,
  LoadingOutlined,
  PlusOutlined,
  SearchOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useI18n } from '@/i18n'
import {
  checkModelConfigName,
  createModelConfig,
  deleteModelConfig,
  fetchAvailableModels,
  listModelConfigs,
  testConnectivity,
  updateModelConfig,
  type ConnectivityTestInput,
  type ModelConfig,
  type ModelConfigInput,
} from '@/api/modelConfig'
import styles from './ModelConfigPage.module.css'

const POLL_INTERVAL_MS = 1500

interface FormValues extends ModelConfigInput {}

type NameCheck = { status: '' | 'validating' | 'success' | 'error'; help?: string }

export default function ModelConfigPage() {
  const { t, locale } = useI18n()
  const { notification } = AntdApp.useApp()
  const [form] = Form.useForm<FormValues>()

  const [data, setData] = useState<ModelConfig[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [searchName, setSearchName] = useState('')
  const [loading, setLoading] = useState(false)
  const [modalOpen, setModalOpen] = useState(false)
  const [editing, setEditing] = useState<ModelConfig | null>(null)
  const [saving, setSaving] = useState(false)
  const [modalTesting, setModalTesting] = useState(false)
  const [rowTesting, setRowTesting] = useState<number | null>(null)
  const [fetching, setFetching] = useState(false)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [nameCheck, setNameCheck] = useState<NameCheck>({ status: '' })
  const debounceTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  const typeValue = Form.useWatch('type', form)
  const autoThinkingOff = Form.useWatch('auto_disable_thinking', form)

  // Typing in the search box fires a request per keystroke; without this a slow
  // earlier response would land last and show results for a stale keyword.
  const loadSeq = useRef(0)

  const load = useCallback(async () => {
    const seq = ++loadSeq.current
    setLoading(true)
    try {
      const res = await listModelConfigs(searchName, page, pageSize)
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

  // Connectivity results arrive as a floating notification that stays 3s.
  function reportTest(ok: boolean, message: string, durationMs: number | null) {
    notification[ok ? 'success' : 'error']({
      message: ok ? t('modelConfig.testSuccess', { ms: durationMs ?? '-' }) : t('modelConfig.testFailed'),
      description: ok ? undefined : message,
      placement: 'topRight',
      duration: 3,
    })
  }

  async function runTest(input: ConnectivityTestInput) {
    try {
      const res = await testConnectivity(input)
      reportTest(res.success, res.message, res.duration_ms)
    } catch (e) {
      reportTest(false, (e as Error).message, null)
    }
  }

  async function onRowTest(record: ModelConfig) {
    setRowTesting(record.id)
    try {
      await runTest({ ...record, api_key: record.api_key ?? null })
    } finally {
      setRowTesting(null)
    }
  }

  function openCreate() {
    setEditing(null)
    setAvailableModels([])
    setNameCheck({ status: '' })
    form.resetFields()
    form.setFieldsValue({
      type: 'llm',
      api_format: 'openai',
      auto_disable_thinking: true,
    } as FormValues)
    setModalOpen(true)
  }

  function openEdit(record: ModelConfig) {
    setEditing(record)
    setAvailableModels([])
    setNameCheck({ status: '' })
    form.setFieldsValue({
      ...record,
      api_key: record.api_key ?? undefined,
      // The box reflects whether a setting is stored: it is what a save clears
      // when it is unticked.
      auto_disable_thinking: record.thinking_state != null,
    } as FormValues)
    setModalOpen(true)
  }

  /** A model is referred to by name afterwards — a testset keeps llm_name, an
   * evaluation the judge's — so a name is checked as it is typed, the way the
   * corpus, testset and adapter dialogs already check theirs. */
  function onNameChange(value: string) {
    setNameCheck({ status: '' })
    clearTimeout(debounceTimer.current)
    const name = value.trim()
    if (!name) return
    const excludeId = editing?.id
    debounceTimer.current = setTimeout(async () => {
      setNameCheck({ status: 'validating' })
      try {
        const res = await checkModelConfigName(name, excludeId)
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

  async function validatedValues(): Promise<FormValues | null> {
    try {
      return await form.validateFields()
    } catch {
      return null
    }
  }

  // "Test" in the dialog validates only the fields the probe needs, so a user
  // can check connectivity before filling in a name.
  async function onModalTest() {
    let values: Pick<FormValues, 'type' | 'api_format' | 'base_url' | 'api_key' | 'model'>
    try {
      values = await form.validateFields(['type', 'api_format', 'base_url', 'api_key', 'model'])
    } catch {
      return
    }
    setModalTesting(true)
    try {
      await runTest({ ...values, api_key: values.api_key ?? null })
    } finally {
      setModalTesting(false)
    }
  }

  async function onSave() {
    const values = await validatedValues()
    if (!values) return
    // The backend rejects a duplicate too (409); checking first is what turns
    // it into the same red hint the other dialogs show while typing.
    try {
      const { available } = await checkModelConfigName(values.name.trim(), editing?.id)
      if (!available) {
        setNameCheck({ status: 'error', help: t('corpus.nameUnavailable', { name: values.name }) })
        return
      }
    } catch {
      setNameCheck({ status: 'error', help: t('corpus.nameCheckFailed') })
      return
    }
    setSaving(true)
    try {
      // Connectivity first, as the dialog promises — and not only because the
      // probe needs a reachable endpoint: a config nobody can call is not worth
      // saving, and saying so now beats finding out on the first run.
      const reachable = await testConnectivity({
        type: values.type,
        api_format: values.api_format,
        base_url: values.base_url,
        api_key: values.api_key ?? null,
        model: values.model,
      })
      if (!reachable.success) {
        notification.error({
          message: t('modelConfig.thinkingNotSaved'),
          description: reachable.message,
          placement: 'topRight',
          duration: 3,
        })
        return
      }
      const willProbe = values.type === 'llm' && values.auto_disable_thinking === true

      const input = { ...values, api_key: values.api_key ?? null }
      // Everything the button set in motion finishes before anything is
      // reported: connectivity, then the probe the save runs, then the record is
      // written. The spinner on the button is what says so — a toast arriving
      // mid-save would arrive behind a dialog nobody has dismissed yet.
      const saved = editing
        ? await updateModelConfig(editing.id, input)
        : await createModelConfig(input)
      setModalOpen(false)
      void load()

      notification.success({
        message: t('modelConfig.testSuccess', { ms: reachable.duration_ms ?? '-' }),
        placement: 'topRight',
        duration: 3,
      })
      // Only an LLM config with the box ticked is probed, so only that one hears
      // about the probe. An embedding — or an LLM whose box was unticked — gets
      // the connectivity result and the save confirmation, nothing else.
      if (willProbe) {
        // Same life as the other two: they are raised together, so a longer one
        // lingers after its neighbours have gone for no reason the reader can
        // see. The verdict's copy is long, but a toast pauses while the pointer
        // rests on it.
        notification[saved.thinking_state ? 'success' : 'warning']({
          message: t('modelConfig.thinkingOff'),
          description: thinkingOutcome(saved),
          placement: 'topRight',
          duration: 3,
        })
      }
      notification.success({
        message: t('modelConfig.saveSuccess'),
        placement: 'topRight',
        duration: 3,
      })
    } catch (e) {
      notification.error({
        message: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
    } finally {
      setSaving(false)
    }
  }

  async function onFetchModels() {
    let values: Pick<FormValues, 'type' | 'api_format' | 'base_url' | 'api_key'>
    try {
      values = await form.validateFields(['type', 'api_format', 'base_url', 'api_key'])
    } catch {
      return
    }
    setFetching(true)
    try {
      const res = await fetchAvailableModels({
        api_format: values.api_format,
        base_url: values.base_url,
        api_key: values.api_key ?? null,
      })
      // Fetched but empty, or not fetched at all: the combobox simply has no
      // dropdown, which is the documented behaviour.
      setAvailableModels(res.models)
    } catch (e) {
      notification.error({
        message: t('modelConfig.fetchFailed'),
        description: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
    } finally {
      setFetching(false)
    }
  }

  async function onDelete(record: ModelConfig) {
    try {
      await deleteModelConfig(record.id)
      notification.success({
        message: t('modelConfig.deleteSuccess'),
        placement: 'topRight',
        duration: 3,
      })
      void load()
    } catch (e) {
      // a 404 (removed in another tab) used to be a silent no-op
      notification.error({
        message: t('common.deleteFailed'),
        description: (e as Error).message,
        placement: 'topRight',
        duration: 3,
      })
    }
  }

  const columns: ColumnsType<ModelConfig> = [
    { title: t('modelConfig.name'), dataIndex: 'name', key: 'name' },
    {
      title: t('modelConfig.type'),
      dataIndex: 'type',
      key: 'type',
      render: (value: ModelConfig['type']) => (value === 'llm' ? 'LLM' : 'Embedding'),
    },
    {
      title: t('modelConfig.apiFormat'),
      dataIndex: 'api_format',
      key: 'api_format',
      render: (value: ModelConfig['api_format']) => (value === 'openai' ? 'OpenAI' : 'Anthropic'),
    },
    // Deliberately not translated: the design doc keeps "Model" as-is.
    { title: 'Model', dataIndex: 'model', key: 'model' },
    {
      title: t('modelConfig.thinkingOff'),
      key: 'thinking_state',
      // Wide enough for the English header ("Thinking off" needs 114px with its
      // padding; 96px wrapped it onto two lines).
      width: 120,
      // Only an embedding is blank. An LLM reads 是 when thinking is actually
      // off (already_off or a setting we probed out) and 否 otherwise — which
      // covers "could not be turned off", "never probed" and "the box was
      // unticked" alike: in all three, nothing is sent and the model may think.
      render: (_: unknown, record: ModelConfig) =>
        record.type !== 'llm'
          ? ''
          : record.thinking_state === 'already_off' || record.thinking_state === 'disabled'
            ? t('common.yes')
            : t('common.no'),
    },
    {
      title: t('modelConfig.actions'),
      key: 'actions',
      render: (_value, record) => (
        <span className={styles.actions}>
          <Button
            type="link"
            size="small"
            icon={<EditOutlined />}
            onClick={() => openEdit(record)}
          >
            {t('modelConfig.edit')}
          </Button>
          <Button
            type="link"
            size="small"
            icon={<ApiOutlined />}
            loading={rowTesting === record.id}
            onClick={() => void onRowTest(record)}
          >
            {t('modelConfig.test')}
          </Button>
          <Popconfirm
            title={t('modelConfig.deleteConfirm')}
            onConfirm={() => void onDelete(record)}
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>
              {t('modelConfig.delete')}
            </Button>
          </Popconfirm>
        </span>
      ),
    },
  ]

  /** What the save-time probe concluded, in the customer's words. */
  function thinkingOutcome(record: ModelConfig): string {
    if (record.thinking_state === 'unsupported') return t('modelConfig.thinkingUnsupported')
    if (record.thinking_state === 'already_off') return t('modelConfig.thinkingAlreadyOff')
    if (record.thinking_state) return t('modelConfig.thinkingDisabled')
    return t('modelConfig.thinkingUnknown')
  }

  const nameHelp = nameCheck.help ? (
    <span className={styles.nameHelp}>
      {nameCheck.status === 'success' && <CheckCircleFilled style={{ color: '#52c41a' }} />}
      {nameCheck.status === 'error' && <CloseCircleFilled style={{ color: '#ff4d4f' }} />}
      <span className={styles.nameHelpText}>{nameCheck.help}</span>
    </span>
  ) : undefined

  return (
    <main className={styles.page}>
      <div className={styles.toolbar}>
        <Input
          allowClear
          prefix={<SearchOutlined />}
          placeholder={t('modelConfig.searchPlaceholder')}
          style={{ width: 280 }}
          onChange={(e) => {
            setPage(1)
            setSearchName(e.target.value)
          }}
        />
        <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
          {t('modelConfig.create')}
        </Button>
      </div>

      <Table
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
        title={t(editing ? 'modelConfig.editTitle' : 'modelConfig.createTitle')}
        open={modalOpen}
        maskClosable={false}
        onCancel={() => setModalOpen(false)}
        footer={[
          <Button key="test" icon={<ApiOutlined />} loading={modalTesting} onClick={() => void onModalTest()}>
            {t('modelConfig.test')}
          </Button>,
          <Button key="cancel" onClick={() => setModalOpen(false)}>
            {t('modelConfig.cancel')}
          </Button>,
          <Button key="save" type="primary" loading={saving} onClick={() => void onSave()}>
            {t('modelConfig.save')}
          </Button>,
        ]}
      >
        {/* "Model Name" is 97px wide in English with its asterisk; 90px cut it. */}
        <Form
          form={form}
          layout="horizontal"
          labelCol={{ flex: locale === 'zh' ? '90px' : '110px' }}
          wrapperCol={{ flex: '1 1 0%' }}
          colon={false}
        >
          <Form.Item
            name="name"
            label={t('modelConfig.name')}
            validateStatus={nameCheck.status}
            help={nameHelp}
            rules={[{ required: true, message: t('modelConfig.nameRequired') }]}
          >
            <Input onChange={(e) => onNameChange(e.target.value)} />
          </Form.Item>

          <Form.Item name="type" label={t('modelConfig.type')} rules={[{ required: true }]}>
            <Radio.Group
              onChange={() => {
                // Anthropic has no embedding models, so switching to Embedding
                // forces the OpenAI format rather than letting the pair be saved.
                if (form.getFieldValue('type') === 'embedding') {
                  form.setFieldValue('api_format', 'openai')
                }
              }}
            >
              <Radio value="llm">LLM</Radio>
              <Radio value="embedding">Embedding</Radio>
            </Radio.Group>
          </Form.Item>

          <Form.Item name="api_format" label={t('modelConfig.apiFormat')} rules={[{ required: true }]}>
            <Radio.Group>
              <Radio value="openai">OpenAI</Radio>
              <Radio value="anthropic" disabled={typeValue === 'embedding'}>
                Anthropic
              </Radio>
            </Radio.Group>
          </Form.Item>

          <Form.Item
            name="base_url"
            label="Base URL"
            rules={[{ required: true, message: t('modelConfig.baseUrlRequired') }]}
          >
            <Input placeholder="https://api.openai.com/v1" autoComplete="off" />
          </Form.Item>

          {/* Optional for every type. Ollama, vLLM and the on-prem endpoints a
              customer actually runs need no key, and one that does is caught by
              the connectivity test the save button runs — which reports the 401
              rather than leaving the form to guess. */}
          <Form.Item name="api_key" label="API Key">
            <Input.Password autoComplete="new-password" />
          </Form.Item>

          <Form.Item label="Model" required>
            <Space.Compact style={{ width: '100%' }}>
              <Form.Item
                name="model"
                noStyle
                rules={[{ required: true, message: t('modelConfig.modelRequired') }]}
              >
                <AutoComplete
                  style={{ width: '100%' }}
                  options={availableModels.map((m) => ({ value: m }))}
                  suffixIcon={availableModels.length > 0 ? <DownOutlined /> : null}
                  filterOption={(input, option) =>
                    (option?.value ?? '').toLowerCase().includes(input.toLowerCase())
                  }
                />
              </Form.Item>
              <Button
                className={styles.fetchBtn}
                disabled={fetching}
                onClick={() => void onFetchModels()}
              >
                {fetching ? <LoadingOutlined /> : t('modelConfig.fetch')}
              </Button>
            </Space.Compact>
          </Form.Item>

          {/* The box the save button reads: ticked means the backend probes this
              endpoint for a way to turn thinking off and stores what it finds;
              unticked means it clears any stored setting instead. The note only
              appears while unticked, as with the other checkboxes in the app. */}
          {typeValue === 'llm' && (
            <Form.Item
              name="auto_disable_thinking"
              valuePropName="checked"
              label=" "
              help={
                autoThinkingOff === false ? (
                  <span className={styles.formNote}>
                    {t('modelConfig.autoThinkingOffNote')}
                  </span>
                ) : undefined
              }
            >
              {/* Same words as the list column and the result toast: one key, so
                  the three can never drift apart. */}
              <Checkbox>{t('modelConfig.thinkingOff')}</Checkbox>
            </Form.Item>
          )}
        </Form>
      </Modal>
    </main>
  )
}
