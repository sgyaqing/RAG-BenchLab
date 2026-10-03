import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import en from './locales/en'
import zh from './locales/zh'
import {
  DEFAULT_LOCALE,
  detectLocale,
  LOCALE_STORAGE_KEY,
  type Locale,
} from '@/utils/locale'

const messages: Record<Locale, typeof en> = { en, zh }

function lookup(obj: unknown, path: string): string {
  const value = path.split('.').reduce<unknown>(
    (acc, key) => (acc && typeof acc === 'object' ? (acc as Record<string, unknown>)[key] : undefined),
    obj,
  )
  return typeof value === 'string' ? value : path
}

export type TParams = Record<string, string | number>
export type TFunc = (key: string, params?: TParams) => string

function interpolate(template: string, params?: TParams): string {
  if (!params) return template
  return template.replace(/\{(\w+)\}/g, (_, name) => String(params[name] ?? `{${name}}`))
}

interface I18nContextValue {
  locale: Locale
  t: TFunc
  setLocale: (locale: Locale) => void
}

const I18nContext = createContext<I18nContextValue>({
  locale: DEFAULT_LOCALE,
  t: (key) => lookup(messages[DEFAULT_LOCALE], key),
  setLocale: () => {},
})

export function I18nProvider({ children }: { children: React.ReactNode }) {
  // Pure SPA: the language can be detected synchronously at startup.
  const [locale, setLocaleState] = useState<Locale>(() =>
    detectLocale(localStorage.getItem(LOCALE_STORAGE_KEY), navigator.language),
  )

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next)
    localStorage.setItem(LOCALE_STORAGE_KEY, next)
  }, [])

  useEffect(() => {
    document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en'
  }, [locale])

  const t: TFunc = (key, params) => interpolate(lookup(messages[locale], key), params)

  return (
    <I18nContext.Provider value={{ locale, t, setLocale }}>
      {children}
    </I18nContext.Provider>
  )
}

export function useI18n() {
  return useContext(I18nContext)
}
