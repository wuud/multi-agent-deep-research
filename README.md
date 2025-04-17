# 多智能体协作式Open Deep Research

## 概述

本项目利用多个专门的 AI 智能体（Agent）协同工作，参考了项目 [OpenDeepResearcher](https://github.com/mshumer/OpenDeepResearcher?tab=readme-ov-file)，根据用户提供的查询，通过网络搜索、内容提取、信息验证、动态摘要和最终报告生成，来完成一个小型研究任务。系统的核心是 **PlannerAgent**，它负责协调整个流程，动态生成搜索策略，并决定何时结束研究。

## 主要特性

*   **多智能体架构**: 将复杂的任务分解给不同的智能体，每个智能体专注于特定功能（规划、搜索、抓取、分析、验证、摘要、报告）。
*   **迭代式研究**: 通过多轮搜索和信息处理，逐步深入和完善研究结果。
*   **动态查询生成**: PlannerAgent 能根据当前研究进展和初步计划，智能地生成后续的搜索查询，或判断研究是否完成。
*   **内容抓取与解析**: 利用 Jina AI RAG API 高效抓取网页内容并提取主要文本。
*   **上下文分析与验证**: AnalyzerAgent 提取与查询相关的信息，ValidatorAgent 进一步评估信息片段的质量和相关性，确保最终报告基于可靠内容。
*   **滚动摘要**: 当收集到的信息过多时，SummarizerAgent 会介入，对累积的上下文进行压缩和整合，防止超出后续处理的限制。
*   **自动化报告**: ReporterAgent 根据所有验证过的上下文，自动生成一份包含引用来源的结构化中文研究报告。
*   **可配置性**:
    *   支持 Serper 和 SerpApi 两种搜索引擎。
    *   可以为不同的 Agent 配置不同的 LLM 模型。
    *   可调整内容处理、上下文长度等多种限制参数。
*   **异步执行**: 利用 `asyncio` 和 `aiohttp` 进行高效的并发网络请求（搜索、抓取）。

## 系统架构与工作流程

系统由以下几个核心 Agent 组成：

1.  **PlannerAgent**: 流程的“大脑”，负责：
    *   接收用户查询和设置。
    *   生成初步研究计划（可选）。
    *   生成初始和后续的搜索查询。
    *   协调其他 Agent 的工作。
    *   管理研究状态（已处理 URL、已用查询、累积上下文）。
    *   判断研究是否完成或达到迭代上限。
    *   触发滚动摘要。
    *   调用 ReporterAgent 生成最终报告。
2.  **SearcherAgent**: 执行网络搜索，使用配置的搜索引擎（Serper/SerpApi）获取相关网页 URL。
3.  **FetcherAgent**: 使用 Jina AI RAG API 抓取指定 URL 的网页内容。
4.  **AnalyzerAgent**: 分析 FetcherAgent 获取的页面内容，提取与用户原始查询和当前搜索查询相关的信息片段。
5.  **ValidatorAgent**: 评估 AnalyzerAgent 提取出的信息片段，判断其是否真实相关、内容充实且非冗余。
6.  **SummarizerAgent**: 当累积的验证后上下文过长时，负责将其与新获取的上下文进行整合与摘要，生成更精简但信息丰富的版本。
7.  **ReporterAgent**: 在研究结束后，整合所有通过验证的上下文片段，生成最终的结构化研究报告，并附带引用来源列表。

**工作流程图:**

```mermaid
graph TD
    A[开始] --> B{输入用户查询 & 迭代次数};
    B --> C[Planner: 初始化];
    C --> D(Planner: 生成初步研究计划);
    D --> E(Planner: 生成初始搜索查询);
    E --> F{开始研究迭代循环};

    F --> G[Planner: 使用当前查询列表];
    G --> H(Searcher: 并发执行搜索);
    H --> I{收集新的、未处理的 URL};
    I -- 有新URL --> J{并发处理链接};
    I -- 无新URL --> N[检查是否需要摘要];

    subgraph 处理单个链接 [For Each New URL]
        J --> K(Fetcher: 抓取页面内容);
        K --> L(Analyzer: 分析内容, 提取上下文);
        L -- 提取到上下文 --> M{Validator: 验证上下文质量};
        L -- 未提取或无关 --> J_End[跳过此URL];
        M -- 验证通过 --> M_OK(存储验证后的上下文 + URL);
        M -- 验证失败 --> J_End;
        M_OK --> J_End;
    end
    J --> J_End;

    J_End --> N;
    N -- 有新验证的上下文 --> O{Planner: 检查累积上下文是否超长};
    N -- 无新验证的上下文 --> P[Planner: 评估是否继续];

    O -- 超长 --> Q(Summarizer: 生成滚动摘要);
    O -- 未超长 --> R(Planner: 追加新上下文到摘要区);
    Q --> P;
    R --> P;

    P -- 迭代次数未满 --> S(Planner: 生成下一轮搜索查询);
    P -- 迭代次数已满 --> T[结束迭代循环];

    S -- 生成新查询 --> G; # 回到循环开始，使用新查询
    S -- LLM决定停止或无新查询 --> T;

    T --> U(Reporter: 整合所有验证过的上下文);
    U --> V(Reporter: 生成最终报告);
    V --> W{输出最终报告和统计信息};
    W --> Z[结束];
    %% Styling (Optional)
    classDef agent fill:#f9f,stroke:#333,stroke-width:2px;
    class C,D,E,G,I,N,O,P,R,S,T,U,V agent;
    classDef external fill:#ccf,stroke:#333,stroke-width:2px;
    class H,K,L,M,Q external;

## 配置

在运行脚本 `py` 之前，请务必修改文件开头的配置常量部分：

### API 密钥

- **SERPAPI_API_KEY**: 如果 `SEARCH_ENGINE` 设置为 `serpapi`，则需要填入您的 SerpApi 密钥。
- **SERPER_API_KEY**: 如果 `SEARCH_ENGINE` 设置为 `serper`（默认），则需要填入您的 Serper API 密钥。
- **JINA_API_KEY**: Jina AI API 密钥（用于 `FetcherAgent`）。
- **OPENAI_API_KEY** 和 **BASE_URL**: 用于初始化 OpenAI client。

### 搜索引擎选择

- **SEARCH_ENGINE**: 设置为 `'serper'` 或 `'serpapi'` 来选择搜索引擎。

### 模型选择

- **DEFAULT_MODEL**: 设置一个默认的 LLM 模型名称。
- **PLANNER_MODEL**, **ANALYZER_MODEL**, **VALIDATOR_MODEL**, **SUMMARIZER_MODEL**, **REPORTER_MODEL**: 可以为特定 Agent 指定模型。如果留空（`""` 或 `None`），则该 Agent 会使用 `DEFAULT_MODEL`。建议为 Planner 和 Validator 选择能力强或成本效益高的模型。

### 内容长度限制

- **MAX_CONTEXT_CHARS_FOR_PLANNER**: 传递给 Planner 判断是否继续或生成查询的摘要最大长度。
- **MAX_CHARS_IN_ROLLING_SUMMARY**: 触发滚动摘要的累积上下文阈值。
- **MAX_CONTEXT_CHARS_FOR_FINAL_REPORT**: Reporter 生成最终报告时能处理的最大上下文长度。
- **MAX_CHARS_PER_PAGE_FOR_ANALYSIS**: Analyzer 处理单个页面内容的最大长度。
- **MAX_CONTEXT_CHARS_FOR_VALIDATION**: Validator 评估单个上下文片段的最大长度。
- **MAX_CHARS_FOR_INITIAL_PLAN**: 生成初始计划时允许的最大输入长度。

# 启动

```bash
python multi-agent_deep_research.py
```
