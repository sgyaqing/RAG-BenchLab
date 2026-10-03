import { useEffect, useRef, useState } from 'react'
import { useI18n } from '@/i18n'
import { SUPPORTED_LOCALES, type Locale } from '@/utils/locale'
import styles from './LanguageSwitch.module.css'

export default function LanguageSwitch() {
  const { locale, t, setLocale } = useI18n()
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function onClickOutside(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('click', onClickOutside)
    return () => document.removeEventListener('click', onClickOutside)
  }, [])

  function switchTo(lang: Locale) {
    setLocale(lang)
    setOpen(false)
  }

  return (
    <div ref={rootRef} className={styles.langSwitch}>
      <button
        className={styles.iconBtn}
        title={t('language.label')}
        aria-label={t('language.label')}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor"
          strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="9" />
          <path d="M3 12h18" />
          <path d="M12 3c2.5 2.4 3.8 5.6 3.8 9S14.5 18.6 12 21c-2.5-2.4-3.8-5.6-3.8-9S9.5 5.4 12 3z" />
        </svg>
      </button>

      {open && (
        <ul className={styles.menu} role="menu">
          {SUPPORTED_LOCALES.map((lang) => (
            <li
              key={lang}
              role="menuitem"
              className={`${styles.menuItem} ${locale === lang ? styles.active : ''}`}
              onClick={() => switchTo(lang)}
            >
              <span className={styles.check} aria-hidden="true">
                {locale === lang ? '✓' : ''}
              </span>
              {t(`language.${lang}`)}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
