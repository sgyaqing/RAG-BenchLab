import { useEffect, useRef } from 'react'
import { Modal } from 'antd'

import { formatDateTime } from '@/utils/format'
import type { LogEntry } from '@/utils/log'
import styles from './LogModal.module.css'

interface Props {
  title: string
  open: boolean
  onClose: () => void
  entries: LogEntry[]
  /** Shown after the lines, unless the entries already carry a failure event. */
  error?: string | null
  /** One entry's copy; null hides the line. See utils/log.ts. */
  textOf: (entry: LogEntry) => string | null
  width?: number
}

/**
 * The log dialog every page shows.
 *
 * There were four of these, one per page, and they drifted apart: three
 * rendered "[time] event" lines and the fourth a summary paragraph, because
 * corpus conversion kept counters instead of events and each page had grown
 * its own frame around whatever it displayed. Sharing the frame is what stops
 * that happening again — the pages differ only in `textOf`, which is the part
 * that genuinely belongs to them.
 */
export default function LogModal({
  title, open, onClose, entries, error, textOf, width = 640,
}: Props) {
  const boxRef = useRef<HTMLDivElement>(null)

  // A long log should open at its end, where the newest line is.
  useEffect(() => {
    if (open && boxRef.current) {
      boxRef.current.scrollTop = boxRef.current.scrollHeight
    }
  }, [open, entries])

  return (
    <Modal
      title={title}
      open={open}
      footer={null}
      // Consistent with every other dialog here: a stray click on the backdrop
      // does not dismiss. Only Esc and the close button do.
      maskClosable={false}
      onCancel={onClose}
      width={width}
    >
      <div ref={boxRef} className={styles.logBox}>
        {entries.map((entry, i) => {
          const text = textOf(entry)
          return text === null ? null : (
            <p className={styles.logLine} key={i}>
              [{formatDateTime(entry.time)}] {text}
            </p>
          )
        })}
        {error && !entries.some((e) => e.key === 'failed') && (
          <p className={styles.logLine}>{error}</p>
        )}
      </div>
    </Modal>
  )
}
