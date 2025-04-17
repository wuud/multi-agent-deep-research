import asyncio
import aiohttp
import json
from openai import AsyncOpenAI, APIError
import re
import sys
import traceback
from typing import List, Dict, Optional, Union, Tuple, Set, Any

# =======================
# 配置常量
# =======================
# --- 用户需要配置 ---
SERPAPI_API_KEY = ""  # 替换为您的 SERPAPI API 密钥
SERPAPI_URL = "https://serpapi.com/search"
SERPER_API_KEY = ""
SERPER_URL = "https://google.serper.dev/search"
SEARCH_ENGINE = "serper"  # Choose 'serper' or 'serpapi'

# JINA  https://jina.ai/api-dashboard/key-manager
JINA_BASE_URL = "https://r.jina.ai/"
JINA_API_KEY = ""  # 替换为您的 JINA API 密钥
# 模型部分
OPENAI_API_KEY = ''  # Replace with your key if needed
BASE_URL = ""  # Replace with your endpoint if needed
# DEFAULT_MODEL = "doubao-1-5-pro-32k-250115"  # 默认使用的模型
DEFAULT_MODEL = "deepseek-v3-250324"  # 默认使用的模型

# 为各 Agent 指定模型。如果留空 ("" or None)，则会使用上面的 DEFAULT_MODEL
PLANNER_MODEL = "deepseek-r1-250120"  # 用于规划、生成查询、判断完成。留空则使用 DEFAULT_MODEL。
ANALYZER_MODEL = ""  # 用于分析页面、提取上下文。留空则使用 DEFAULT_MODEL。
VALIDATOR_MODEL = "doubao-1-5-lite-32k-250115"  # 用于验证上下文质量。留空则使用 DEFAULT_MODEL。
SUMMARIZER_MODEL = ""  # 用于按需摘要。留空则使用 DEFAULT_MODEL。
REPORTER_MODEL = ""  # 用于生成最终报告。留空则使用 DEFAULT_MODEL。

MAX_CONTEXT_CHARS_FOR_PLANNER = 30000  # 传递给 Planner 判断是否继续的摘要长度限制
MAX_CHARS_IN_ROLLING_SUMMARY = int(MAX_CONTEXT_CHARS_FOR_PLANNER * 0.9)  # 触发摘要的阈值
MAX_CONTEXT_CHARS_FOR_FINAL_REPORT = 30000  # 最终报告上下文限制
MAX_CHARS_PER_PAGE_FOR_ANALYSIS = 15000  # 分析器处理页面内容限制
MAX_CONTEXT_CHARS_FOR_VALIDATION = 5000  # 验证器处理单个片段限制
MAX_CHARS_FOR_INITIAL_PLAN = 4000  # 初始计划提示的 LLM 输入限制

# ============================
# 初始化 OpenAI 客户端
# ============================
try:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
    print("OpenAI 客户端初始化成功。")
except Exception as e:
    print(f"初始化 OpenAI 客户端时出错: {e}")
    print(f"请确保 OPENAI_API_KEY 和 BASE_URL 已正确设置。")
    client = None


# ============================
# 通用 LLM 调用函数
# ============================
async def call_llm_async(messages: List[Dict[str, str]], model: str) -> Optional[str]:
    """
    通用的异步 LLM 调用函数。
    注意：这里的 model 参数应该是已经确定好的具体模型名称。
    """
    if not client:
        print("LLM 客户端未初始化。")
        return None
    if not model:
        print("错误：尝试调用 LLM 时模型名称为空。")
        return None
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            # request_timeout=60 # 可选超时
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            print(f"意外的 LLM 响应结构 (模型: {model}):", response)
            return None
    except APIError as e:
        print(f"LLM API 错误 (模型: {model}): Status={e.status_code}, Message={e.message}, Type={e.type}")
        # Handle specific rate limit errors if needed
        # if e.status_code == 429:
        #     print("触发速率限制，将等待...")
        #     await asyncio.sleep(5) # Example wait
        return None
    except Exception as e:
        print(f"调用 LLM API 时出错 (模型: {model}): {e}")
        return None


# ============================
# Agent 类定义 (Searcher, Fetcher, Analyzer, Validator, Summarizer, Reporter)
# ============================

class SearcherAgent:
    """负责执行网络搜索（支持 SerpApi 和 Serper）"""

    async def search(self, session: aiohttp.ClientSession, query: str) -> List[str]:
        """使用配置的搜索引擎执行搜索并返回 URL 列表"""
        links = []
        engine_to_use = SEARCH_ENGINE.lower()

        if engine_to_use == "serper":
            # print(f"SearcherAgent - 使用 Serper 搜索: {query}") # Reduced verbosity
            if not SERPER_API_KEY or SERPER_API_KEY.startswith("YOUR_"):
                print("警告: SearcherAgent - SERPER_API_KEY 未配置或无效，无法使用 Serper 进行搜索。")
                return []

            payload = json.dumps({"q": query, "num": 10})
            headers = {
                'X-API-KEY': SERPER_API_KEY,
                'Content-Type': 'application/json'
            }
            try:
                async with session.post(SERPER_URL, headers=headers, data=payload, timeout=20) as resp:
                    if resp.status == 200:
                        try:
                            results = await resp.json()
                            links = [item.get("link") for item in results.get("organic", []) if item.get("link")]
                            links = [link for link in links if link and link.startswith(('http://', 'https://'))]
                            # print(f"SearcherAgent - Serper 找到 {len(links)} 个链接。")
                        except (aiohttp.ContentTypeError, json.JSONDecodeError) as json_err:
                            print(f"SearcherAgent - Serper 响应 JSON 解析错误 (查询: {query}): {json_err}")
                            # print(f"原始响应文本: {await resp.text()}") # Optional debug
                    else:
                        print(f"SearcherAgent - Serper API 错误 (查询: {query}): {resp.status} - {await resp.text()}")
            except asyncio.TimeoutError:
                print(f"SearcherAgent - Serper 搜索超时 (查询: {query})。")
            except Exception as e:
                print(f"SearcherAgent - Serper 搜索过程中发生意外错误 (查询: {query}): {e}")

        elif engine_to_use == "serpapi":
            # print(f"SearcherAgent - 使用 SerpApi 搜索: {query}") # Reduced verbosity
            if not SERPAPI_API_KEY or SERPAPI_API_KEY.startswith("YOUR_"):
                print("警告: SearcherAgent - SERPAPI_API_KEY 未配置或无效，无法使用 SerpApi 进行搜索。")
                return []

            params = {"q": query, "api_key": SERPAPI_API_KEY, "engine": "google", "num": 10}
            try:
                async with session.get(SERPAPI_URL, params=params, timeout=20) as resp:
                    if resp.status == 200:
                        try:
                            results = await resp.json()
                            links = [item.get("link") for item in results.get("organic_results", []) if
                                     item.get("link")]
                            links = [link for link in links if link and link.startswith(('http://', 'https://'))]
                            # print(f"SearcherAgent - SerpApi 找到 {len(links)} 个链接。")
                        except (aiohttp.ContentTypeError, json.JSONDecodeError) as json_err:
                            print(f"SearcherAgent - SerpApi 响应 JSON 解析错误 (查询: {query}): {json_err}")
                            # print(f"原始响应文本: {await resp.text()}") # Optional debug
                    else:
                        print(f"SearcherAgent - SerpApi API 错误 (查询: {query}): {resp.status} - {await resp.text()}")
            except asyncio.TimeoutError:
                print(f"SearcherAgent - SerpApi 搜索超时 (查询: {query})。")
            except Exception as e:
                print(f"SearcherAgent - SerpApi 搜索过程中发生意外错误 (查询: {query}): {e}")
        else:
            print(f"错误: SearcherAgent - 未知的 SEARCH_ENGINE 配置: '{SEARCH_ENGINE}'。请设置为 'serpapi' 或 'serper'。")

        return links


class FetcherAgent:
    """负责抓取网页内容"""

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> str:
        """使用 Jina API 抓取 URL 内容"""
        if not JINA_API_KEY or JINA_API_KEY == "YOUR_JINA_API_KEY":
            print(f"警告: FetcherAgent - JINA_API_KEY 未配置 (URL: {url})。")
            return ""

        full_url = f"{JINA_BASE_URL}{url}"
        headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Accept": "application/json",
                   "X-With-Generated-Alt": "false"}
        try:
            async with session.get(full_url, headers=headers, timeout=45) as resp:
                if resp.status == 200:
                    try:
                        data = await resp.json()
                        content = data.get('data', {}).get('content', '')
                        if content:
                            content = re.sub(r'\s{3,}', '\n\n', content).strip()
                            print(f"FetcherAgent - 成功获取上下文: {url} (上下文长度: {len(content)})")
                            return content
                        else:
                            print(f"FetcherAgent - Jina 返回空内容 ({url})。")
                            return ""  # Return empty string if content is empty
                    except (aiohttp.ContentTypeError, json.JSONDecodeError):
                        text_content = await resp.text()
                        print(f"FetcherAgent - Jina 响应非预期格式 ({url})。返回原始文本 (长度: {len(text_content)})")
                        return text_content  # Return raw text if JSON fails
                else:
                    print(f"FetcherAgent - Jina 获取出错 ({url}): {resp.status} - {await resp.text()}")
                    return ""
        except asyncio.TimeoutError:
            print(f"FetcherAgent - Jina 获取超时 ({url})。")
            return ""
        except aiohttp.ClientConnectorError as conn_err:
            print(f"FetcherAgent - Jina 连接错误 ({url}): {conn_err}")
            return ""
        except Exception as e:
            print(f"FetcherAgent - Jina 获取网页出错 ({url}): {e}")
            traceback.print_exc()  # Print stack trace for unexpected errors
            return ""


class AnalyzerAgent:
    """负责分析页面内容，判断相关性并提取信息"""

    async def analyze(self, session: aiohttp.ClientSession, user_query: str, search_query: str, page_content: str,
                      url: str) -> Optional[str]:
        """分析页面内容，如果相关则提取上下文"""
        if not page_content or page_content.isspace():
            # print(f"  AnalyzerAgent - 跳过空内容: {url}") # Less verbose
            return None

        prompt = (
            "你是一位专业的信息提取员。根据用户的原始查询、用于找到此页面的搜索查询"
            "以及下面的网页内容，提取所有与回答用户原始查询直接相关的信息片段。"
            "专注于关键事实、数据和见解。避免无关信息、导航链接、页眉页脚、广告等非实质性内容。"
            "如果内容与查询无关，请返回空字符串或“无关”。返回纯文本提取物。"  # Added instruction for irrelevant content
        )
        truncated_content = page_content[:MAX_CHARS_PER_PAGE_FOR_ANALYSIS] + (
            "..." if len(page_content) > MAX_CHARS_PER_PAGE_FOR_ANALYSIS else "")
        messages = [
            {"role": "system", "content": "你是提取和总结相关信息的专家，专注于实质性内容。"},
            {"role": "user",
             "content": (f"用户原始查询: {user_query}\n"
                         f"搜索查询: {search_query}\n\n"
                         f"网页内容 (来自 {url}, 最多 {MAX_CHARS_PER_PAGE_FOR_ANALYSIS} 字符):\n---\n{truncated_content}\n---\n\n"
                         f"{prompt}")
             }
        ]
        model_to_use = ANALYZER_MODEL or DEFAULT_MODEL
        extracted_context = await call_llm_async(messages, model=model_to_use)

        if extracted_context and not extracted_context.strip().lower() in ["", "无关", "irrelevant", "n/a", "none"]:
            clean_context = extracted_context.strip()
            print(f"  AnalyzerAgent - 已提取上下文来自: {url} (使用模型: {model_to_use}, 长度: {len(clean_context)})")
            return clean_context
        else:
            # print(f"  AnalyzerAgent - 未能从页面提取相关上下文或内容无关: {url} (使用模型: {model_to_use})") # Less verbose
            return None


class ValidatorAgent:
    """负责验证提取的上下文片段的质量和相关性"""

    async def validate(self, session: aiohttp.ClientSession, user_query: str, search_query: str, extracted_context: str,
                       url: str) -> bool:
        """验证上下文片段是否高质量且与搜索查询紧密相关"""
        if not extracted_context:
            return False

        prompt = (
            "你是一位严谨的研究验证员。请评估以下从特定网页提取的【上下文片段】。"
            "判断这个片段是否：\n"
            "1. **高度相关**: 与【用户原始查询】直接相关？\n"
            "2. **内容充实**: 包含具体信息、事实或数据，而不仅仅是通用陈述、意见或导航链接？\n"
            "3. **非冗余**: 提供了与查询相关的新信息，而不仅仅是重复已知信息？\n"
            "请综合考虑以上三点，如果都满足，请回答 'Yes'，否则回答 'No'。只回答 'Yes' 或 'No'。"
        )
        truncated_context = extracted_context[:MAX_CONTEXT_CHARS_FOR_VALIDATION] + (
            "..." if len(extracted_context) > MAX_CONTEXT_CHARS_FOR_VALIDATION else "")
        messages = [
            {"role": "system", "content": "你是一位严格、注重细节的研究验证员。只回答 'Yes' 或 'No'。"},
            {"role": "user",
             "content": (f"用户原始查询: {user_query}\n"
                         f"搜索查询: {search_query}\n"
                         f"来源 URL: {url}\n\n"
                         f"【上下文片段】 (最多 {MAX_CONTEXT_CHARS_FOR_VALIDATION} 字符):\n{truncated_context}\n\n"
                         f"{prompt}")
             }
        ]
        model_to_use = VALIDATOR_MODEL or DEFAULT_MODEL
        response = await call_llm_async(messages, model=model_to_use)

        if response:
            answer = response.strip().capitalize()
            if answer.startswith("Yes"):
                print(f"  ValidatorAgent - 上下文通过验证: {url} (使用模型: {model_to_use})")
                return True
            elif answer.startswith("No"):
                print(f"  ValidatorAgent - 上下文未通过验证: {url} (使用模型: {model_to_use})")
                return False
            else:
                print(f"  ValidatorAgent - 验证响应意外 ({url}, 模型 {model_to_use}): {response} -> 视为无效")
                return False
        print(f"  ValidatorAgent - 验证调用失败 ({url}) -> 视为无效")
        return False


class SummarizerAgent:
    """负责按需进行滚动摘要"""

    async def summarize(self, session: aiohttp.ClientSession, current_summary: str,
                        new_context_text: str, user_query: str) -> str:
        """在需要时压缩和整合信息"""
        if not new_context_text:
            return current_summary

        # Truncate new context if it alone exceeds limit (unlikely but safe)
        truncated_new_context = new_context_text[:MAX_CHARS_IN_ROLLING_SUMMARY] + (
            "..." if len(new_context_text) > MAX_CHARS_IN_ROLLING_SUMMARY else "")

        prompt_overhead = 1000  # Estimate tokens for prompt instructions
        available_space_for_summary = MAX_CONTEXT_CHARS_FOR_PLANNER - len(truncated_new_context) - prompt_overhead
        if available_space_for_summary < 0:
            available_space_for_summary = 0

        truncated_current_summary = current_summary
        if len(current_summary) > available_space_for_summary:
            truncated_current_summary = "...\n" + current_summary[-available_space_for_summary:]
            print(f"  SummarizerAgent - 截断现有摘要以适应 LLM 输入 (保留最后 {available_space_for_summary} 字符)")

        prompt = (
            "你是一位高效的研究摘要员。由于累积的研究信息过长，需要进行压缩。"
            "请将【本轮新验证发现】整合到【现有摘要精华】中，生成一份更新后的、简洁但信息丰富的【综合摘要】。"
            "保留关键信息点和最新进展，去除冗余，紧密围绕【用户原始查询】。"
            "只输出更新后的【综合摘要】文本。"
        )
        messages = [
            {"role": "system", "content": "你是一位专注于在信息超载时进行压缩和整合、生成简洁研究摘要的专家。"},
            {"role": "user",
             "content": (f"用户原始查询: {user_query}\n\n"
                         f"【现有摘要精华】(可能从开头截断):\n{truncated_current_summary}\n\n"
                         f"【本轮新验证发现】(最多 {MAX_CHARS_IN_ROLLING_SUMMARY} 字符):\n{truncated_new_context}\n\n"
                         f"{prompt}")
             }
        ]
        model_to_use = SUMMARIZER_MODEL or DEFAULT_MODEL
        print(f"  SummarizerAgent - 上下文超长，正在调用 LLM ({model_to_use}) 进行滚动摘要...")
        updated_summary = await call_llm_async(messages, model=model_to_use)

        if updated_summary:
            final_summary = updated_summary.strip()
            print(f"  SummarizerAgent - 滚动摘要压缩完成 (新长度: {len(final_summary)})。")
            return final_summary
        else:
            print("  SummarizerAgent - 错误：LLM 未能生成滚动摘要。将尝试保留现有摘要的最后部分。")
            # Fallback: keep the most recent part of the old summary + new contexts (truncated if needed)
            fallback_summary = current_summary + "\n\n" + new_context_text
            if len(fallback_summary) > MAX_CHARS_IN_ROLLING_SUMMARY:
                return fallback_summary[-MAX_CHARS_IN_ROLLING_SUMMARY:]
            else:
                return fallback_summary


class ReporterAgent:
    """负责生成最终报告"""

    async def generate_report(self, session: aiohttp.ClientSession, user_query: str,
                              validated_contexts_with_urls: List[Dict[str, Any]]) -> str:
        """根据所有验证过的上下文生成最终报告"""
        if not validated_contexts_with_urls:
            return "未能收集到足够经过验证的信息来生成报告。"

        unique_urls = sorted(list(set(item['url'] for item in validated_contexts_with_urls)))
        url_to_index = {url: i + 1 for i, url in enumerate(unique_urls)}

        context_for_prompt = ""
        for item in validated_contexts_with_urls:
            source_index = url_to_index[item['url']]
            context_for_prompt += f"来源 [{source_index}] ({item['url']}):\n{item['context']}\n\n---\n\n"

        # Smart truncation: keep more recent contexts if truncation is needed
        if len(context_for_prompt) > MAX_CONTEXT_CHARS_FOR_FINAL_REPORT:
            print(
                f"ReporterAgent - 警告: 用于生成报告的总上下文长度 ({len(context_for_prompt)}) 超过限制 ({MAX_CONTEXT_CHARS_FOR_FINAL_REPORT})，将从开头截断。")
            # Rebuild context string from the end until limit is reached (approximately)
            truncated_context_list = []
            current_len = 0
            prompt_overhead = 1000  # Reserve space for prompt
            target_len = MAX_CONTEXT_CHARS_FOR_FINAL_REPORT - prompt_overhead

            for item in reversed(validated_contexts_with_urls):
                source_index = url_to_index[item['url']]
                item_text = f"来源 [{source_index}] ({item['url']}):\n{item['context']}\n\n---\n\n"
                if current_len + len(item_text) <= target_len:
                    truncated_context_list.append(item_text)
                    current_len += len(item_text)
                else:
                    # Try adding a partial context if it's the first one we encounter that overflows
                    if not truncated_context_list:
                        remaining_space = target_len - current_len
                        partial_text = item_text[:remaining_space] + "...\n\n---\n\n"
                        truncated_context_list.append(partial_text)
                    break  # Stop adding more contexts

            context_for_prompt = "...\n\n---\n\n" + "".join(reversed(truncated_context_list))
        else:
            # No truncation needed case handled implicitly
            pass

        prompt = (
            "你是一位专业的研究员和报告撰写人。根据用户的原始查询以及下面从多个来源收集并经过验证的上下文信息（每个来源已编号），"
            "撰写一份全面、结构良好、详细的中文报告，彻底回答该查询。"
            "报告应综合信息，流畅呈现，包含相关见解和结论。"
            "在报告正文中，使用方括号引用来源编号（例如 `[1]`）。"
            "报告最后必须包含名为“引用来源”的部分，列出所有编号对应的 URL。\n"
            "不要包含关于此提示或你角色的评论。确保报告语言流畅、专业。"  # Added fluency requirement
        )
        messages = [
            {"role": "system", "content": "你是一位熟练的报告撰写人，注重准确性、引用和中文表达的流畅性。"},
            {"role": "user",
             "content": (f"用户原始查询: {user_query}\n\n"
                         f"收集并验证的相关上下文 (可能从开头截断以适应限制):\n{context_for_prompt}\n\n"  # Updated description
                         f"来源 URL 列表 (供报告末尾引用):\n" +
                         "\n".join([f"[{i + 1}] {url}" for i, url in enumerate(unique_urls)]) +
                         f"\n\n{prompt}")
             }
        ]
        model_to_use = REPORTER_MODEL or DEFAULT_MODEL
        report = await call_llm_async(messages, model=model_to_use)

        if report:
            final_report = report.strip()
            # Check and append references if missing (more robust check)
            if not re.search(r"引用来源|References|Sources", final_report, re.IGNORECASE):
                print(f"ReporterAgent - 警告：最终报告似乎缺少引用部分 (使用模型: {model_to_use})。正在追加...")
                references_section = "\n\n==== 引用来源 ====\n" + "\n".join(
                    f"[{i + 1}] {url}" for i, url in enumerate(unique_urls))
                final_report += references_section
            return final_report
        else:
            # Fallback message
            error_msg = f"ReporterAgent - 无法生成最终报告 (使用模型: {model_to_use})。"
            if validated_contexts_with_urls:
                error_msg += "\n已收集到以下经过验证的来源：\n" + "\n".join(
                    f"- {item['url']}" for item in validated_contexts_with_urls)
            return error_msg


class PlannerAgent:
    """负责整个研究流程的规划、协调和状态管理"""

    def __init__(self, iteration_limit: int = 3):
        self.iteration_limit = iteration_limit
        self.user_query: str = ""
        # 状态管理
        self.research_plan: Optional[str] = None  # <<< NEW: To store the initial plan
        self.validated_contexts_with_urls: List[Dict[str, Any]] = []
        self.current_research_summary: str = ""
        self.all_search_queries_used: Set[str] = set()
        self.processed_urls: Set[str] = set()

        # 初始化其他 Agent
        self.searcher = SearcherAgent()
        self.fetcher = FetcherAgent()
        self.analyzer = AnalyzerAgent()
        self.validator = ValidatorAgent()
        self.summarizer = SummarizerAgent()
        self.reporter = ReporterAgent()

    async def generate_initial_plan(self, session: aiohttp.ClientSession) -> Optional[str]:
        """根据用户查询生成初步的研究计划"""
        model_to_use = PLANNER_MODEL or DEFAULT_MODEL
        prompt = (
            "你是一位细致的研究规划师。基于以下用户查询，创建一个结构化的研究计划。"
            "将主要查询分解为 5-7 个关键的、逻辑顺序的子问题或研究领域。"
            "将计划呈现为一个清晰的、编号的中文列表，每个点都是一个具体的研究步骤或要调查的问题。"
            "例如:\n"
            "1. 定义核心概念及其背景。\n"
            "2. 调查相关的主要方法或技术。\n"
            "3. 分析关键的挑战或争议点。\n"
            "...\n"
            "只输出编号的计划列表，不要添加任何额外的解释或引言。"
        )
        messages = [
            {"role": "system", "content": "你是一位研究规划专家，擅长将复杂问题分解为结构化的研究步骤。"},
            {"role": "user", "content": f"用户查询:\n```\n{self.user_query}\n```\n\n{prompt}"}
        ]

        plan = await call_llm_async(messages, model=model_to_use)
        if plan:
            return plan.strip()
        else:
            print(f"PlannerAgent - 未能生成初始研究计划 (使用模型: {model_to_use})。")
            return None

    async def generate_search_queries(self, session: aiohttp.ClientSession, iteration, iteration_limit,
                                      initial: bool = False) -> Union[str, List[str]]:
        """生成初始查询或根据当前状态生成新查询"""

        # Check iteration limit early
        if iteration >= iteration_limit:  # Use >= for clarity
            print(f'PlannerAgent - 已达到最大迭代次数 ({self.iteration_limit})，停止生成新查询。')
            return []

        model_to_use = PLANNER_MODEL or DEFAULT_MODEL

        if initial:
            prompt = (
                "你是一位专业的研究助手。根据用户的原始查询，生成最多四个不同的、精确的初始搜索查询，用于开始研究。"
                "这些查询应旨在获取关于查询主题的概述性信息和关键方面。"
                "只返回 Python 列表格式，例如：['query1', 'query2']。"
                f"参考以下研究计划：\n{self.research_plan}\n\n"
            )
            messages = [
                {"role": "system", "content": "你是一个有帮助且精确的研究助手，专注于生成有效的初始搜索查询。"},
                {"role": "user", "content": f"用户原始查询: {self.user_query}\n\n{prompt}"}
            ]
        else:
            truncated_summary = self.current_research_summary[:MAX_CONTEXT_CHARS_FOR_PLANNER] + \
                                ("..." if len(self.current_research_summary) > MAX_CONTEXT_CHARS_FOR_PLANNER else "")
            prompt = (
                "你是一位分析型研究策略师。根据【用户原始查询】、【之前已使用的搜索查询】和【当前研究摘要】，"
                "评估研究是否已足够全面地回答了用户查询，或者是否还需要探索新的角度。\n"
                "1. 如果研究已足够全面，或者摘要显示已找到充分信息，请准确回复 `<done>`。\n"
                "2. 如果需要进一步研究，请生成最多四个新的、具体的、与【之前已使用的搜索查询】不同的搜索查询，以填补信息空白或探索未覆盖的方面。"
                "   只返回 Python 列表格式，例如：['new query 1', 'new query 2']。\n"
                "重点关注那些尚未在摘要中充分体现的方面。\n"
                "只输出 Python 列表或标记 `<done>`。"
            )
            messages = [
                {"role": "system", "content": "你是一个系统的研究规划师和评估员，专注于决定何时停止以及生成后续查询。"},
                {"role": "user",
                 "content": (f"用户原始查询: {self.user_query}\n"
                             f"之前已使用的搜索查询: {list(self.all_search_queries_used)}\n\n"
                             f"当前研究摘要 (最多 {MAX_CONTEXT_CHARS_FOR_PLANNER} 字符):\n{truncated_summary}\n\n"
                             # Optional: Could add self.research_plan here too
                             f"初始研究计划供参考：\n{self.research_plan}\n\n"
                             f"{prompt}")
                 }
            ]

        response = await call_llm_async(messages, model=model_to_use)

        if response:
            cleaned = response.strip()
            if cleaned == "<done>":
                print(f"PlannerAgent - LLM 评估完成，决定停止搜索 (模型: {model_to_use})。")
                return "<done>"
            try:
                match = re.search(r'```python\s*(\[.*?\])\s*```|(\[.*?\])', cleaned, re.DOTALL)
                if match:
                    new_queries_str = match.group(1) or match.group(2)
                    new_queries_str = new_queries_str.replace('`', '')  # Clean potential backticks

                    import ast
                    evaluated_list = ast.literal_eval(new_queries_str)

                    if isinstance(evaluated_list, list) and all(isinstance(q, str) for q in evaluated_list):
                        # Filter out empty strings and already used queries
                        filtered_queries = [q.strip() for q in evaluated_list if
                                            q.strip() and q.strip() not in self.all_search_queries_used]
                        if not filtered_queries:
                            print(f"PlannerAgent - LLM 生成了查询，但都是重复或空的 (模型: {model_to_use})。停止。")
                            return []
                        print(
                            f"PlannerAgent - LLM 生成了 {len(filtered_queries)} 个新查询 (模型: {model_to_use}): {filtered_queries[:4]}")
                        return filtered_queries[:4]  # Limit to max 4
                    else:
                        print(
                            f"PlannerAgent - LLM 未返回有效查询列表 (解析后类型: {type(evaluated_list)}) (模型: {model_to_use}): {response}")
                        return []
                else:
                    print(f"PlannerAgent - 响应中找不到查询列表（非 '<done>'） (模型: {model_to_use}): {response}")
                    return []
            except (SyntaxError, ValueError) as e:
                print(f"PlannerAgent - 解析查询列表时出错 (模型: {model_to_use}): {e}\n原始响应: {response}")
                return []
            except Exception as e:
                print(f"PlannerAgent - 解析查询时发生意外错误 (模型: {model_to_use}): {e}\n响应: {response}")
                return []  # Catch other potential eval errors
        else:
            print(f"PlannerAgent - 生成查询的 LLM 调用失败或返回空 (模型: {model_to_use})。")
            return []  # LLM call failed

    async def process_single_link(self, session: aiohttp.ClientSession, url: str, search_query: str) -> Optional[
        Dict[str, Any]]:
        """处理单个链接的完整流程：抓取 -> 分析 -> 验证"""
        if url in self.processed_urls:
            return None

        print(f"  PlannerAgent - 开始处理链接: {url} (来自查询: '{search_query}')")
        content = await self.fetcher.fetch(session, url)
        self.processed_urls.add(url)  # Mark as processed even if fetch fails, to avoid retries

        if not content:
            print(f"  PlannerAgent - 抓取失败或无内容: {url}")
            return None

        extracted_context = await self.analyzer.analyze(session, self.user_query, search_query, content, url)
        if not extracted_context:
            print(f"  PlannerAgent - 分析器未提取到相关内容: {url}")
            return None

        is_valid = await self.validator.validate(session, self.user_query, search_query, extracted_context, url)
        if is_valid:
            return {'context': extracted_context, 'url': url, 'search_query': search_query}
        else:
            return None

    async def run(self, user_query: str):
        """执行整个研究流程"""
        if not client:
            print("无法启动，LLM 客户端未初始化。")
            return
        self.user_query = user_query.strip()
        if not self.user_query:
            print("用户查询不能为空。")
            return

        print(f"\n===== 开始研究: {self.user_query} =====")
        print(f"最大迭代次数: {self.iteration_limit}")
        print(
            f"使用的模型: Planner={PLANNER_MODEL or DEFAULT_MODEL}, Analyzer={ANALYZER_MODEL or DEFAULT_MODEL}, Validator={VALIDATOR_MODEL or DEFAULT_MODEL}, Summarizer={SUMMARIZER_MODEL or DEFAULT_MODEL}, Reporter={REPORTER_MODEL or DEFAULT_MODEL}")
        print(f"搜索引擎: {SEARCH_ENGINE}")
        print("========================================\n")

        async with aiohttp.ClientSession() as session:
            # ----- 初始规划 -----
            print("[阶段 1/5] PlannerAgent - 生成初步研究计划...")
            self.research_plan = await self.generate_initial_plan(session)
            if self.research_plan:
                print("\n初步研究计划:\n----------")
                print(self.research_plan)
                print("----------\n")
            else:
                print("PlannerAgent - 未能生成初步研究计划，将直接生成搜索查询。\n")

            # ----- 初始查询 -----
            print("[阶段 2/5] PlannerAgent - 生成初始搜索查询...")

            initial_queries = await self.generate_search_queries(session, 0, self.iteration_limit, initial=True)

            if not isinstance(initial_queries, list) or not initial_queries:
                print("PlannerAgent - 未能生成有效的初始搜索查询。退出。")
                return
            print(f"PlannerAgent - 生成的初始查询: {initial_queries}")
            queries_for_this_iteration = initial_queries
            self.all_search_queries_used.update(q.strip() for q in queries_for_this_iteration)  # Add stripped queries

            # ----- 迭代研究 -----
            print("\n[阶段 3/5] PlannerAgent - 开始迭代研究...")
            iteration = 0
            while iteration < self.iteration_limit:
                print(
                    f"\n===== 研究迭代 {iteration + 1} / {self.iteration_limit} =====")  # Use iteration+1 for 1-based display
                if not queries_for_this_iteration:
                    print("PlannerAgent - 没有需要执行的查询，跳过此迭代。")
                    break
                print(f"PlannerAgent - 使用查询: {queries_for_this_iteration}")

                # --- 并发搜索 ---
                search_tasks = [self.searcher.search(session, query) for query in queries_for_this_iteration]
                search_results_per_query = await asyncio.gather(*search_tasks)

                # --- 收集本轮要处理的 URL ---
                links_to_process_this_iteration: List[Tuple[str, str]] = []
                unique_urls_found_this_round = set()
                for i, urls in enumerate(search_results_per_query):
                    if i < len(queries_for_this_iteration):  # Safety check
                        query = queries_for_this_iteration[i]
                        for url in urls:
                            if url not in self.processed_urls and url not in unique_urls_found_this_round:
                                links_to_process_this_iteration.append((url, query))
                                unique_urls_found_this_round.add(url)
                    else:
                        print(
                            f"警告：搜索结果数量 ({len(search_results_per_query)}) 与查询数量 ({len(queries_for_this_iteration)}) 不匹配")

                print(f"\nPlannerAgent - 找到 {len(links_to_process_this_iteration)} 个新的、唯一的链接进行处理...")

                if not links_to_process_this_iteration:
                    print("PlannerAgent - 在此迭代中没有新的链接需要处理。")
                    new_validated_contexts_this_iter = []  # Ensure this is defined
                else:
                    # --- 并发处理链接 ---
                    processing_tasks = [self.process_single_link(session, url, sq) for url, sq in
                                        links_to_process_this_iteration]
                    results = await asyncio.gather(*processing_tasks)
                    new_validated_contexts_this_iter = [res for res in results if res is not None]
                    print(
                        f"\nPlannerAgent - 在此迭代中获得 {len(new_validated_contexts_this_iter)} 个通过验证的新上下文片段。")

                # --- 更新摘要和状态 ---
                if new_validated_contexts_this_iter:
                    self.validated_contexts_with_urls.extend(new_validated_contexts_this_iter)
                    new_context_text_for_check = ""
                    for item in new_validated_contexts_this_iter:
                        new_context_text_for_check += f"来源: {item['url']}\n内容:\n{item['context']}\n\n---\n\n"

                    if len(self.current_research_summary) + len(
                            new_context_text_for_check) > MAX_CHARS_IN_ROLLING_SUMMARY:
                        self.current_research_summary = await self.summarizer.summarize(
                            session, self.current_research_summary, new_context_text_for_check, self.user_query
                        )
                    else:
                        print("  PlannerAgent - 上下文长度未超阈值，直接追加新内容到摘要区。")
                        self.current_research_summary += new_context_text_for_check
                        print(f"  PlannerAgent - 当前累积摘要区长度: {len(self.current_research_summary)}")

                elif iteration > 0:  # Only print if not the first iteration and no new context found
                    print("PlannerAgent - 在此迭代中未找到新的有用上下文。")

                # Increment iteration counter *before* generating next queries
                iteration += 1

                # ----- 判断是否需要下一轮 -----
                print(f"\nPlannerAgent - 评估是否需要进一步搜索 (迭代 {iteration}/{self.iteration_limit})...")
                # Pass the *current* iteration number (which is now iteration+1 logically)
                next_step = await self.generate_search_queries(session, iteration, self.iteration_limit, initial=False)

                if next_step == "<done>":
                    print("PlannerAgent - 决定停止研究。")
                    queries_for_this_iteration = []  # Set to empty to break loop
                    break  # Exit loop explicitly
                elif isinstance(next_step, list) and next_step:
                    # print(f"PlannerAgent - LLM 提供了新的搜索查询: {next_step}") # Already printed inside generate_search_queries
                    queries_for_this_iteration = next_step
                    self.all_search_queries_used.update(
                        q.strip() for q in queries_for_this_iteration)  # Add stripped queries
                else:
                    # This case includes empty list [], LLM failure, or non-list/non-"<done>" response
                    if isinstance(next_step, list) and not next_step:
                        print("PlannerAgent - LLM 未提供新的有效查询。结束搜索。")
                    # else: (error cases already printed inside generate_search_queries)
                    queries_for_this_iteration = []  # Set to empty to break loop
                    break  # Exit loop explicitly

                # Optional short delay between iterations if needed
                # await asyncio.sleep(1)

            # End of while loop (iterations complete or stopped)

            # ----- 生成报告 -----
            print("\n=========================")
            print("[阶段 4/5] PlannerAgent - 研究完成。调用 ReporterAgent 生成最终报告...")
            print("=========================")
            if not self.validated_contexts_with_urls:
                print("\nPlannerAgent - 警告：未能收集到任何经过验证的相关上下文。报告可能为空或基于无信息。")

            final_report = await self.reporter.generate_report(session, self.user_query,
                                                               self.validated_contexts_with_urls)

            # ----- 输出结果 -----
            print("\n\n========== 最终报告 ==========\n")
            print(final_report)
            print("\n===============================\n")

            # Stage 5: Print final summary info
            print(f"[阶段 5/5] 研究流程结束。")
            print(f"总共处理了 {len(self.processed_urls)} 个 URL。")
            print(
                f"总共使用了 {len(self.all_search_queries_used)} 个独特的搜索查询: {list(self.all_search_queries_used)}")
            if self.validated_contexts_with_urls:
                unique_urls_final = sorted(list(set(item['url'] for item in self.validated_contexts_with_urls)))
                print(
                    f"报告基于 {len(self.validated_contexts_with_urls)} 个验证过的上下文片段，引用了 {len(unique_urls_final)} 个独特的 URL:")
                # Optionally print URLs again, or rely on report's reference section
                # for url in unique_urls_final:
                #     print(f"- {url}")
            else:
                print("报告未基于任何验证过的上下文片段。")


# =========================
# 主函数入口
# =========================
def main():
    """脚本入口"""
    if client is None:
        print("\n错误：OpenAI 客户端未能初始化，程序无法运行。")
        print("请检查您的 OPENAI_API_KEY 和 BASE_URL 设置。")
        return

    user_query = ""
    # Example query for quick testing:
    # user_query = "大模型知识冲突"
    while not user_query:
        user_query_input = input("请输入您的研究查询/主题 (或直接回车使用示例 '大模型知识冲突'): ").strip()
        if not user_query_input:
            user_query = "大模型知识冲突"
            print(f"使用示例查询: {user_query}")
        else:
            user_query = user_query_input
        # user_query = input("请输入您的研究查询/主题: ").strip()
        if not user_query:
            print("查询不能为空，请重新输入。")

    iter_limit_input = input("输入最大迭代次数（默认为 3）: ").strip()
    try:
        iteration_limit = int(iter_limit_input)
        if iteration_limit <= 0:
            print("迭代次数必须大于 0，使用默认值 3。")
            iteration_limit = 3
    except ValueError:
        print("输入无效，使用默认迭代次数 3。")
        iteration_limit = 3

    # 创建并运行 PlannerAgent
    planner = PlannerAgent(iteration_limit=iteration_limit)
    try:
        asyncio.run(planner.run(user_query))
    except KeyboardInterrupt:
        print("\n操作被用户中断。")
    except Exception as e:
        print(f"\n在主程序执行过程中发生意外错误: {e}")
        print("错误追踪:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
