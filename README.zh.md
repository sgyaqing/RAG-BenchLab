<p align="center"><a href="README.md">English</a> | <a href="README.zh.md">中文</a></p>

# <img src="assets/logo.png" width="38" alt=""> RAG-BenchLab

快速、智能、开箱即用的 RAG 系统评测工具：**从你自己的语料生成测试集（问答对），再用业内公认的指标给你的 RAG 系统打分。**

![架构与使用示意](assets/architecture.png)

## 功能与亮点

**一站式 RAG 系统检测**
建测试集、评测 RAG 系统，一个工具完成。

**简单易用，无需写代码**
全程在界面上操作，无需了解复杂知识，不写代码。

**高质量测试集生成**
自动质检筛查，输出高质量测试集（问答对），减少人工修改。

**快速、精准的 RAG 系统评测**
内置五个专业 RAG 评测指标。智能缓存，在 RAG 系统调参的多次调用场景减少 token 成本。

**智能、全面的 RAG 系统适配**
智能对接各种 RAG 系统。

## 使用方法

启动后浏览器打开 <http://localhost:6742>，按顺序走五步：

**1. 配置模型**
添加一个 LLM 和一个 Embedding：填 Base URL、API Key、Model，点「保存」。

**2. 配置 RAG 适配器**（只生成测试集则不需要配置）
告诉本工具怎么调你的 RAG 系统，两种方式：

- **智能填写** —— 填平台名（如 `RAGFlow`）或该平台的 API 文档地址，自动推出接口配置并真机验证
- **手动填写** —— 自己填请求头、请求体模板和回答 / 上下文提取路径

**3. 建文集**
上传你的文档（zip 或目录或文档文件）。

**4. 生成测试集**
选文集、模型，设置生成参数，等它生成问答对。生成完可以逐条查看、修改、删除，也可以导出或导入。

**5. 评测 RAG 系统**
选测试集、RAG 适配器、模型，勾选指标，跑完看报告。报告包含 RAG 系统的总分以及每个问题的得分。

## 评测指标说明

指标由 [ragas](https://github.com/vibrantlabsai/ragas) 计算，下面的定义取自 ragas 0.4.3。

**Context Precision（上下文精确率）** —— [官方文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)
衡量检索回来的内容里，**相关的是不是排在前面** —— 它算的是排序的平均精确率，不只是"检索到的东西相不相关"。

**Context Recall（上下文召回率）** —— [官方文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
以参考答案为准，看**该检索到的内容有没有被检索到**（用命中与漏检估算）。

**Faithfulness（忠实度）** —— [官方文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
把系统回答拆成一条条说法，逐条判断能不能**从检索到的内容里直接推断出来**。分数 = 有依据的说法数 ÷ 总说法数。分数低，说明回答里有编造、或超出了检索到的内容。

**Response Relevancy（回答相关性）** —— [官方文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)
衡量回答与问题的相关程度。**答非所问、信息不全、啰嗦冗余**都会被扣分。0 到 1，1 最好。

**Answer Correctness（回答正确性）** —— [官方文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_correctness/)
把系统回答和参考答案比对，是**事实一致性**与**语义相似度**的综合分，默认权重 0.75 / 0.25。

> 其中 **Context Precision、Context Recall、Faithfulness** 需要 RAG 系统返回检索到的片段。如果适配器没配置上下文提取路径，这三项会自动置灰，报告里对应的均分也会留空。

## 各平台下安装方法

### Docker（macOS / Windows / Linux 通用）

```bash
IMAGE=sgyaqing/rag-benchlab:0.1.0
#IMAGE=registry.cn-hangzhou.aliyuncs.com/sgyaqing/rag-benchlab:0.1.0

docker run -d --name rag-benchlab \
  -p 6742:6742 \
  -v rbl-data:/app/data \
  -v rbl-logs:/app/logs \
  "$IMAGE"
```

打开 <http://localhost:6742>。停止：`docker stop rag-benchlab`。

> **如果你的 RAG 系统跑在同一台机器上**，Linux 下再加一个参数，容器里的 `localhost` 才能指到宿主机：
> `--add-host=host.docker.internal:host-gateway`
> （macOS 和 Windows 的 Docker Desktop 已经支持，不用加。）

> ⚠️ **本工具当前 v0.1.0 版本没有登录鉴权。** 任何能访问到这个系统的人，都能看到你存在里面的 API Key。请只在内网使用，或放在需要鉴权的反向代理之后，**不要直接暴露到公网。**

### macOS（推荐）

**方式一：绿色版**
从 [Releases](https://github.com/sgyaqing/RAG-BenchLab/releases) 下载 `RAG-BenchLab-0.1.0-macos-arm64.zip`，解压后双击「start.command」。自带 Python 和全部依赖，不需要预先安装任何环境。

> **所有运行方式中，最推荐在 macOS 上用绿色版**：包里的 NumPy 直接调用 Apple 为 macOS 优化的数学库，向量计算快得多，而在 Docker、Windows 下无法使用这层优化。**建测试集和评测的提速很明显。**

**方式二：Docker**
见上面的 Docker 一节。

### Windows

**方式一：绿色版**
从 [Releases](https://github.com/sgyaqing/RAG-BenchLab/releases) 下载 `RAG-BenchLab-0.1.0-windows-x64.zip`，解压后双击「start.bat」。自带 Python，不需要预先安装任何环境。

**方式二：Docker**
见上面的 Docker 一节。

### Linux

见上面的 Docker 一节。

---

## 许可证

Apache License 2.0，版权归大连秦腾科技有限公司所有。详见 [LICENSE](LICENSE)。

本项目包含来自 [ragas](https://github.com/vibrantlabsai/ragas) 的衍生内容，详见 [NOTICE](NOTICE)。
