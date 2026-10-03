/**
 * Timestamps as the UI shows them.
 *
 * The backend stores UTC without a timezone marker, so a bare ISO string has to
 * be told it is UTC before it is displayed in the reader's zone. Four components
 * carried their own copy of this; one is enough.
 */
export function formatDateTime(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(/[zZ]|[+-]\d{2}:?\d{2}$/.test(iso) ? iso : iso + 'Z')
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}
