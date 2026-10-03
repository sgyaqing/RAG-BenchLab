import { useEffect, useRef, useState } from 'react'
import { App as AntdApp, Button, Form, Input, Modal, Spin, Upload } from 'antd'
import { CheckCircleFilled, CloseCircleFilled, InboxOutlined } from '@ant-design/icons'
import { useI18n } from '@/i18n'
import { checkTestsetName, importTestset } from '@/api/testset'
import styles from './CreateCorpusModal.module.css'

type NameCheck = { status: '' | 'validating' | 'success' | 'error'; help?: string }

interface Props {
  open: boolean
  onClose: () => void
  onCreated: () => void
}

const NAME_PATTERN = /^[\w\- 一-鿿]+$/u

export default function ImportTestsetModal({ open, onClose, onCreated }: Props) {
  const { t, locale } = useI18n()
  const { notification } = AntdApp.useApp()
  const [form] = Form.useForm<{ name: string }>()

  const [file, setFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState('')
  const [nameCheck, setNameCheck] = useState<NameCheck>({ status: '' })
  const [importing, setImporting] = useState(false)
  const debounceTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  // Reset when the modal opens (not on close — resetting before the close
  // animation finishes visibly snaps fields back to defaults).
  useEffect(() => {
    if (open) reset()
  }, [open])

  function reset() {
    form.resetFields()
    setFile(null)
    setFileError('')
    setNameCheck({ status: '' })
    setImporting(false)
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

  /** Taken before the name round trip below: that trip is long enough for a
   * second click to start a second import, which would send the file twice. */
  const submitting = useRef(false)

  async function onImport() {
    if (submitting.current) return
    submitting.current = true
    try {
      let name: string
      try {
        name = (await form.validateFields()).name.trim()
      } catch {
        return
      }
      if (!file) {
        setFileError(t('testset.fileRequired'))
        return
      }
      // Re-verify the name right before importing.
      try {
        const res = await checkTestsetName(name)
        if (!res.available) {
          setNameCheck({ status: 'error', help: t('corpus.nameUnavailable', { name }) })
          return
        }
      } catch {
        setNameCheck({ status: 'error', help: t('corpus.nameCheckFailed') })
        return
      }
      setImporting(true)
      try {
        await importTestset(name, file)
        notification.success({
          message: t('testset.importSuccess'),
          placement: 'topRight',
          duration: 3,
        })
        onCreated()
      } catch (e) {
        notification.error({
          message: t('testset.importFailed'),
          description: (e as Error).message,
          placement: 'topRight',
          duration: 5,
        })
        setImporting(false)
      }
    } finally {
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
      title={t('testset.importTitle')}
      open={open}
      maskClosable={false}
      onCancel={() => {
        if (!importing) onClose()
      }}
      footer={[
        <Button key="cancel" disabled={importing} onClick={() => onClose()}>
          {t('testset.cancel')}
        </Button>,
        <Button
          key="import"
          type="primary"
          loading={importing}
          disabled={nameCheck.status === 'error'}
          onClick={() => void onImport()}
        >
          {t('testset.createAction')}
        </Button>,
      ]}
    >
      {/* "Testset Name" is 104px wide in English with its asterisk; 90px cut it. */}
      <Form
        form={form}
        className={styles.form}
        layout="horizontal"
        labelCol={{ flex: locale === 'zh' ? '90px' : '110px' }}
        wrapperCol={{ flex: '1 1 0%' }}
        colon={false}
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
          <Input
            maxLength={64}
            disabled={importing}
            onChange={(e) => onNameChange(e.target.value)}
          />
        </Form.Item>

        <Form.Item label={t('testset.fileLabel')} required>
          <Upload.Dragger
            accept=".jsonl"
            beforeUpload={(f) => {
              // Collect the file manually; upload happens on "Confirm".
              setFile(f)
              setFileError('')
              return Upload.LIST_IGNORE
            }}
            showUploadList={false}
            disabled={importing}
          >
            <p className={styles.pickerIcon}>
              <InboxOutlined />
            </p>
            <p className={styles.pickerText}>{t('testset.importUploadClick')}</p>
            <p className={styles.pickerHint}>{t('testset.importUploadHint')}</p>
          </Upload.Dragger>
          {file && (
            <p className={styles.selected}>
              {t('testset.importSelectedFile', { name: file.name })}
              {'  '}
              <Button
                type="link"
                size="small"
                disabled={importing}
                onClick={() => setFile(null)}
              >
                {t('corpus.clearSelection')}
              </Button>
            </p>
          )}
          {fileError && <p className={styles.error}>{fileError}</p>}
        </Form.Item>

        {importing && (
          <Form.Item wrapperCol={{ offset: 0 }} label=" ">
            <Spin size="small" /> {t('testset.importing')}
          </Form.Item>
        )}
      </Form>
    </Modal>
  )
}
