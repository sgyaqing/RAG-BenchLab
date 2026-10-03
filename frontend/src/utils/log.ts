import type { TFunc, TParams } from '@/i18n'

/**
 * One stored event. The four entities each declared this shape separately and
 * identically; it lives here now, and their api modules alias it.
 */
export interface LogEntry {
  time: string
  key: string
  params: Record<string, string | number | null>
}

/**
 * One log line's text, or null when the locale table has no copy for the key.
 *
 * Every entity names its log keys the same way — `started`, `queryDone`,
 * `judgeCache` — against its own namespace, so one mapping serves them all.
 *
 * Null rather than the raw key: a bundle can outlive a key it does not know
 * (stale build, key added later), and showing `testset.logFoo` to a user is
 * worse than showing nothing. Null-valued params are dropped for the same
 * reason — `{count}` with a null would print "null".
 */
export function logEntryText(
  t: TFunc,
  namespace: string,
  entry: LogEntry,
  decorate?: (params: TParams) => TParams,
): string | null {
  const params = Object.fromEntries(
    Object.entries(entry.params ?? {}).filter(([, v]) => v != null),
  ) as TParams
  const key = `${namespace}.log${entry.key[0].toUpperCase()}${entry.key.slice(1)}`
  const text = t(key, decorate ? decorate(params) : params)
  return text === key ? null : text
}
