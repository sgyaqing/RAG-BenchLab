import { useEffect, useRef, useState, type DragEvent } from 'react'
import { App as AntdApp, Button, Form, Input, Modal, Progress, Space } from 'antd'
import { CheckCircleFilled, CloseCircleFilled, InboxOutlined } from '@ant-design/icons'
import { useI18n } from '@/i18n'
import {
  checkCorpusName,
  createCorpus,
  deleteCorpus,
  processCorpus,
  uploadCorpusFile,
} from '@/api/corpus'
import { analyzePaths, isAllowedUpload, topDir, type SelectionMode } from '@/utils/files'
import { canPickDirectory, pickDirectory } from '@/utils/directory'
import { filesFromDataTransfer } from '@/utils/drop'
import styles from './CreateCorpusModal.module.css'

interface PickedFile {
  file: File
  path: string
}

type NameCheck = { status: '' | 'validating' | 'success' | 'error'; help?: string }

interface Props {
  open: boolean
  onClose: () => void
  onCreated: () => void
}

const NAME_PATTERN = /^[\w\- 一-鿿]+$/

export default function CreateCorpusModal({ open, onClose, onCreated }: Props) {
  const { t, locale } = useI18n()
  const { notification } = AntdApp.useApp()
  const [form] = Form.useForm<{ name: string }>()

  const [files, setFiles] = useState<PickedFile[]>([])
  const [dirTotal, setDirTotal] = useState(0) // raw file count of the picked directory (unfiltered)
  const [nameCheck, setNameCheck] = useState<NameCheck>({ status: '' })
  const [selectionError, setSelectionError] = useState('')
  const [uploading, setUploading] = useState<{ done: number; total: number } | null>(null)
  const debounceTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const dirInputRef = useRef<HTMLInputElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  // dragenter/dragleave fire for every child element as well; counting keeps
  // the highlight steady instead of flickering as the pointer crosses them.
  const dragDepth = useRef(0)

  const mode: SelectionMode = analyzePaths(files.map((f) => f.path))

  function conflictMessage(): string {
    if (mode === 'zip') return t('corpus.conflictZip')
    if (mode === 'dir') return t('corpus.conflictDir')
    return t('corpus.conflictFiles', { count: files.length })
  }

  function reset() {
    form.resetFields()
    setFiles([])
    setDirTotal(0)
    setNameCheck({ status: '' })
    setSelectionError('')
    setUploading(null)
  }

  function onNameChange(value: string) {
    setNameCheck({ status: '' })
    clearTimeout(debounceTimer.current)
    const name = value.trim()
    if (!name || !NAME_PATTERN.test(name)) return
    debounceTimer.current = setTimeout(async () => {
      setNameCheck({ status: 'validating' })
      try {
        const res = await checkCorpusName(name)
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

  function pick(list: File[]) {
    const picked: PickedFile[] = list
      .map((file) => ({
        file,
        path: (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name,
      }))
      // Unsupported files are silently skipped; zip is allowed only standalone.
      .filter((p) => isAllowedUpload(p.path))
    if (picked.length === 0) return

    const newMode = analyzePaths(picked.map((p) => p.path))

    if (mode === 'empty') {
      if (newMode === 'invalid') {
        setSelectionError(t('corpus.onlySingle'))
        return
      }
      setFiles(picked)
      // For a directory, remember the raw (unfiltered) file count for display.
      setDirTotal(newMode === 'dir' ? list.length : 0)
      setSelectionError('')
      return
    }
    // Picking another zip replaces the current one.
    if (mode === 'zip' && newMode === 'zip') {
      setFiles(picked.slice(-1))
      setSelectionError('')
      return
    }
    // Loose files merge with loose files (deduplicated by path).
    if (mode === 'files' && newMode === 'files') {
      const seen = new Set(files.map((f) => f.path))
      setFiles([...files, ...picked.filter((p) => !seen.has(p.path))])
      setSelectionError('')
      return
    }
    // Anything else conflicts with the current selection: reject with a hint.
    setSelectionError(conflictMessage())
  }

  /** Dropping is the one gesture that covers files, a zip and a folder. */
  async function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    const list = await filesFromDataTransfer(e.dataTransfer)
    if (list.length > 0) pick(list)
  }

  function onDragEnter(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    dragDepth.current += 1
    setDragging(true)
  }

  function onDragLeave() {
    dragDepth.current -= 1
    if (dragDepth.current <= 0) {
      dragDepth.current = 0
      setDragging(false)
    }
  }

  function onSelectDir() {
    // Chrome and Edge get the system's folder chooser, which is the only
    // dialog on those two that can return a folder — the panel
    // webkitdirectory opens there is the file panel, where a folder can only
    // be walked into. Every other browser uses that input.
    if (canPickDirectory()) {
      void pickDirectory().then((list) => {
        if (list.length > 0) pick(list)
      })
      return
    }
    dirInputRef.current?.click()
  }

  /** Taken before the name round trip below: that trip is long enough for a
   * second click to create the corpus twice and upload every file twice. */
  const submitting = useRef(false)

  async function onCreate() {
    if (submitting.current) return
    submitting.current = true
    try {
      let name: string
      try {
        const values = await form.validateFields()
        name = values.name.trim()
      } catch {
        return
      }
      if (files.length === 0) {
        setSelectionError(t('corpus.filesRequired'))
        return
      }
      if (mode === 'invalid') {
        setSelectionError(t('corpus.onlySingle'))
        return
      }
      // Re-verify the name right before creating.
      try {
        const res = await checkCorpusName(name)
        if (!res.available) {
          setNameCheck({ status: 'error', help: t('corpus.nameUnavailable', { name }) })
          return
        }
      } catch {
        setNameCheck({ status: 'error', help: t('corpus.nameCheckFailed') })
        return
      }

      let corpusId: number | null = null
      setUploading({ done: 0, total: files.length })
      try {
        const corpus = await createCorpus(name)
        corpusId = corpus.id
        for (let i = 0; i < files.length; i++) {
          await uploadCorpusFile(corpusId, files[i].file, files[i].path)
          setUploading({ done: i + 1, total: files.length })
        }
        await processCorpus(corpusId)
      } catch (e) {
        if (corpusId !== null) {
          await deleteCorpus(corpusId).catch(() => {})
        }
        notification.error({
          message: t('corpus.uploadFailed'),
          description: (e as Error).message,
          placement: 'topRight',
          duration: 3,
        })
        setUploading(null)
        return
      }
      onCreated()
    } finally {
      submitting.current = false
    }
  }

  // Reset when the modal opens (not on close — resetting before the close
  // animation finishes visibly snaps fields back to defaults).
  useEffect(() => {
    if (open) reset()
  }, [open])

  const isUploading = uploading !== null

  const nameHelp = nameCheck.help ? (
    <span className={styles.nameHelp}>
      <span style={{ color: nameCheck.status === 'success' ? '#52c41a' : '#ff4d4f' }}>
        {nameCheck.status === 'success' ? <CheckCircleFilled /> : <CloseCircleFilled />}
      </span>
      <span className={styles.nameHelpText}>{nameCheck.help}</span>
    </span>
  ) : undefined

  return (
    <Modal
      title={t('corpus.createTitle')}
      open={open}
      maskClosable={false}
      onCancel={() => {
        if (!isUploading) {
          onClose()
        }
      }}
      footer={[
        <Button
          key="cancel"
          disabled={isUploading}
          onClick={() => {
            onClose()
          }}
        >
          {t('corpus.cancel')}
        </Button>,
        <Button
          key="create"
          type="primary"
          loading={isUploading}
          disabled={nameCheck.status === 'error'}
          onClick={() => void onCreate()}
        >
          {t('corpus.createAction')}
        </Button>,
      ]}
    >
      {/* "Corpus Name" is 105px wide in English with its asterisk, and the
          label column clips instead of wrapping, so 90px cut the N off. */}
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
          label={t('corpus.nameLabel')}
          validateStatus={nameCheck.status}
          help={nameHelp}
          rules={[
            { required: true, message: t('corpus.nameRequired') },
            { pattern: NAME_PATTERN, message: t('corpus.nameInvalid') },
          ]}
        >
          <Input maxLength={64} onChange={(e) => onNameChange(e.target.value)} />
        </Form.Item>

        <Form.Item label={t('corpus.filesLabel')} required>
          {/* One area, one thing: drop here. Choosing is the two buttons below
              it, one per picker the browser actually has — a dialog returns
              files or a folder, never both, so each button says which it is.
              The area itself opens nothing: when it did, one corner of it
              opened the file panel and another opened the folder chooser, and
              nothing on screen said which was which. */}
          <div
            className={`${styles.dropZone}${dragging ? ` ${styles.dropZoneActive}` : ''}`}
            onDragEnter={onDragEnter}
            onDragOver={(e) => e.preventDefault()}
            onDragLeave={onDragLeave}
            onDrop={(e) => void onDrop(e)}
          >
            <p className={styles.pickerIcon}>
              <InboxOutlined />
            </p>
            <p className={styles.pickerText}>{t('corpus.dropHere')}</p>
            <p className={styles.pickerHint}>{t('corpus.uploadHint')}</p>
            <Space size={8}>
              <Button disabled={isUploading} onClick={() => fileInputRef.current?.click()}>
                {t('corpus.selectFile')}
              </Button>
              <Button disabled={isUploading} onClick={onSelectDir}>
                {t('corpus.selectDir')}
              </Button>
            </Space>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              style={{ display: 'none' }}
              onChange={(e) => {
                const list = Array.from(e.target.files ?? [])
                if (list.length > 0) pick(list)
                e.target.value = ''
              }}
            />
            <input
              ref={dirInputRef}
              type="file"
              style={{ display: 'none' }}
              {...({ webkitdirectory: '' } as Record<string, string>)}
              onChange={(e) => {
                const list = Array.from(e.target.files ?? [])
                if (list.length > 0) pick(list)
                e.target.value = ''
              }}
            />
          </div>
          {files.length > 0 && (
            <p className={styles.selected}>
              {mode === 'zip' && t('corpus.selectedZip', { name: files[0].path })}
              {mode === 'dir' &&
                t('corpus.selectedDir', { name: topDir(files[0].path) ?? '', count: dirTotal })}
              {mode === 'files' && t('corpus.selectedFiles', { count: files.length })}
              {'  '}
              <Button
                type="link"
                size="small"
                disabled={isUploading}
                onClick={() => {
                  setFiles([])
                  setDirTotal(0)
                  setSelectionError('')
                }}
              >
                {t('corpus.clearSelection')}
              </Button>
            </p>
          )}
          {selectionError && <p className={styles.error}>{selectionError}</p>}
        </Form.Item>

        {isUploading && (
          <Form.Item wrapperCol={{ offset: 0 }} label=" ">
            <Progress
              className={styles.uploadProgress}
              percent={Math.round((uploading.done / uploading.total) * 100)}
              format={() => t('corpus.uploading', uploading)}
            />
          </Form.Item>
        )}
      </Form>
    </Modal>
  )
}
