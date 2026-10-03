import { App as AntdApp, ConfigProvider } from 'antd'
import enUS from 'antd/locale/en_US'
import zhCN from 'antd/locale/zh_CN'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { useI18n } from '@/i18n'
import Layout from '@/components/Layout'
import HomePage from '@/components/HomePage'
import ModelConfigPage from '@/components/ModelConfigPage'
import WorkspaceLayout from '@/components/WorkspaceLayout'
import CorpusPage from '@/components/CorpusPage'
import TestsetPage from '@/components/TestsetPage'
import EditTestsetPage from '@/components/EditTestsetPage'
import EvaluationPage from '@/components/EvaluationPage'
import EvalReportPage from '@/components/EvalReportPage'
import RagSystemPage from '@/components/RagSystemPage'

export default function App() {
  const { locale } = useI18n()

  return (
    <ConfigProvider locale={locale === 'zh' ? zhCN : enUS}>
      <AntdApp>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<HomePage />} />
              <Route element={<WorkspaceLayout />}>
                <Route path="/model-configs" element={<ModelConfigPage />} />
                <Route path="/rag-adapters" element={<RagSystemPage />} />
                <Route path="/corpora" element={<CorpusPage />} />
                <Route path="/testsets" element={<TestsetPage />} />
                <Route path="/testsets/:id/edit" element={<EditTestsetPage />} />
                <Route path="/evaluations" element={<EvaluationPage />} />
                <Route path="/evaluations/:id" element={<EvalReportPage />} />
              </Route>
            </Route>
          </Routes>
        </BrowserRouter>
      </AntdApp>
    </ConfigProvider>
  )
}
