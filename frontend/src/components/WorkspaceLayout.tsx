import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import { Menu } from 'antd'
import {
  ApiOutlined,
  EditOutlined,
  ExperimentOutlined,
  FileAddOutlined,
  PlayCircleOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import { useI18n } from '@/i18n'
import styles from './WorkspaceLayout.module.css'

export default function WorkspaceLayout() {
  const { t, locale } = useI18n()
  const navigate = useNavigate()
  const location = useLocation()

  // The edit page is reached via the "Edit" action of a testset row.
  const isEditPage = /^\/testsets\/\d+\/edit/.test(location.pathname)
  // The report page belongs to the "create test" menu entry.
  const isReportPage = /^\/evaluations\/\d+/.test(location.pathname)
  const selectedKey = isEditPage
    ? '/testsets/edit'
    : isReportPage
      ? '/evaluations'
      : location.pathname

  return (
    <div className={styles.container}>
      <aside className={`${styles.sidebar} ${locale === 'en' ? styles.sidebarEn : ''}`}>
        <Menu
          mode="inline"
          selectedKeys={[selectedKey]}
          items={[
            {
              key: '/model-configs',
              icon: <SettingOutlined />,
              label: t('menu.configModel'),
            },
            {
              key: '/rag-adapters',
              icon: <ApiOutlined />,
              label: t('menu.configRagSystem'),
            },
            { type: 'divider' },
            {
              type: 'group',
              label: t('home.buildDataset'),
              children: [
                {
                  key: '/corpora',
                  icon: <FileAddOutlined />,
                  label: t('dataset.createCorpus'),
                },
                {
                  key: '/testsets',
                  icon: <ExperimentOutlined />,
                  label: t('dataset.createTestset'),
                },
                // Only visible while editing a testset (entered via the row action).
                ...(isEditPage
                  ? [
                      {
                        key: '/testsets/edit',
                        icon: <EditOutlined />,
                        label: t('dataset.editTestset'),
                      },
                    ]
                  : []),
              ],
            },
            {
              type: 'group',
              label: t('home.evaluateRag'),
              children: [
                {
                  key: '/evaluations',
                  icon: <PlayCircleOutlined />,
                  label: t('evaluation.createTest'),
                },
              ],
            },
          ]}
          onClick={({ key }) => {
            if (key === '/testsets/edit') return
            navigate(key)
          }}
        />
      </aside>
      <section className={styles.content}>
        <Outlet />
      </section>
    </div>
  )
}
