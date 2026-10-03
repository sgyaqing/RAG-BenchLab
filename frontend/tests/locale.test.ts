import { describe, expect, it } from 'vitest'
import { detectLocale, normalizeLocale } from '../src/utils/locale'
import en from '../src/i18n/locales/en'
import zh from '../src/i18n/locales/zh'

describe('normalizeLocale', () => {
  it('maps Chinese browser languages to zh', () => {
    expect(normalizeLocale('zh-CN')).toBe('zh')
    expect(normalizeLocale('zh-TW')).toBe('zh')
    expect(normalizeLocale('zh')).toBe('zh')
  })

  it('falls back to en for non-Chinese languages', () => {
    expect(normalizeLocale('en-US')).toBe('en')
    expect(normalizeLocale('ja-JP')).toBe('en')
    expect(normalizeLocale(null)).toBe('en')
    expect(normalizeLocale(undefined)).toBe('en')
  })
})

describe('detectLocale', () => {
  it('prefers a valid stored locale over browser language', () => {
    expect(detectLocale('en', 'zh-CN')).toBe('en')
    expect(detectLocale('zh', 'en-US')).toBe('zh')
  })

  it('ignores invalid stored values', () => {
    expect(detectLocale('fr', 'zh-CN')).toBe('zh')
  })

  it('uses browser language when nothing is stored', () => {
    expect(detectLocale(null, 'zh-Hans-CN')).toBe('zh')
    expect(detectLocale(null, 'de-DE')).toBe('en')
  })
})

describe('locale messages', () => {
  it('zh and en expose the same keys', () => {
    const keys = (obj: object, prefix = ''): string[] =>
      Object.entries(obj).flatMap(([k, v]) =>
        typeof v === 'object' && v !== null
          ? keys(v as object, `${prefix}${k}.`)
          : [`${prefix}${k}`],
      )
    expect(keys(zh).sort()).toEqual(keys(en).sort())
  })
})
