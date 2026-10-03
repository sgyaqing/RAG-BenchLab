export type Locale = 'zh' | 'en'

export const SUPPORTED_LOCALES: Locale[] = ['zh', 'en']
export const DEFAULT_LOCALE: Locale = 'en'
export const LOCALE_STORAGE_KEY = 'rag-benchlab-locale'

export function normalizeLocale(raw: string | null | undefined): Locale {
  if (raw && raw.toLowerCase().startsWith('zh')) {
    return 'zh'
  }
  return DEFAULT_LOCALE
}

export function detectLocale(
  stored: string | null | undefined,
  browserLanguage: string | null | undefined,
): Locale {
  if (stored && (SUPPORTED_LOCALES as string[]).includes(stored)) {
    return stored as Locale
  }
  return normalizeLocale(browserLanguage)
}
