import { useNavigate } from 'react-router-dom'
import { useI18n } from '@/i18n'
import styles from './HomePage.module.css'

export default function HomePage() {
  const { t } = useI18n()
  const navigate = useNavigate()

  return (
    <main className={styles.home}>
      <h1 className={styles.title}>{t('app.title')}</h1>
      <p className={styles.tagline}>{t('app.tagline')}</p>

      <div className={styles.entries}>
        <button className={styles.entryBtn} onClick={() => navigate('/corpora')}>
          {t('home.buildDataset')}
        </button>
        <button className={styles.entryBtn} onClick={() => navigate('/evaluations')}>
          {t('home.evaluateRag')}
        </button>
      </div>
    </main>
  )
}
