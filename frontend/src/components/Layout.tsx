import { Link, Outlet } from 'react-router-dom'
import { useI18n } from '@/i18n'
import LanguageSwitch from './LanguageSwitch'
import styles from './Layout.module.css'

export default function Layout() {
  const { t } = useI18n()

  return (
    <div className={styles.layout}>
      <header className={styles.topbar}>
        <Link to="/" className={styles.brand}>
          <img src="/logo.svg" alt="RAG BenchLab logo" width={36} height={36} />
          <span className={styles.brandName}>{t('app.title')}</span>
        </Link>
        <div className={styles.actions}>
          <LanguageSwitch />
        </div>
      </header>
      <Outlet />
    </div>
  )
}
