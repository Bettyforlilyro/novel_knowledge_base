# 语义分块 + 阶段边界检测
import json
import re
from typing import List

from pydantic import BaseModel, Field

from src.core.llm.llm_client import QwenClient
from src.core.settings import get_settings


class ChunkConfig(BaseModel):
    """分块配置模型（对应 chunk_config.yaml 的 chunking 节点）"""
    max_tokens_per_chunk: int = Field(..., description="单个 chunk 最大 token 数")
    min_tokens_per_chunk: int = Field(..., description="单个 chunk 最小 token 数（避免过小）")
    paragraph_separator: str = Field(..., description="段落分隔符，通常为双换行")
    safe_break_keywords: List[str] = Field(..., description="安全断点正则模式（避免切断）")
    use_llm_for_refinement: bool = Field(default=False, description="是否启用 LLM 辅助断点判断")


class TextChunk(BaseModel):
    """分块结果数据结构"""
    start_chapter_idx: int = Field(..., description="分块起始章节索引")
    end_chapter_idx: int = Field(..., description="分块结束章节索引")
    text: str = Field(..., description="分块文本内容")
    estimated_tokens: int = Field(..., description="估计的分块 tokens 数")
    is_natural_break: bool = Field(..., description="是否为剧情自然断点切分")
    break_reason: str = Field(..., description="分块断点原因，如章节结尾、关键词匹配等")


def is_safe_break_point(paragraph: str, config: ChunkConfig) -> bool:
    """
    判断某一段落是否是安全的剧情断点。

    安全断点特征：
    - 包含时间/空间切换关键词（如“数日后”）
    - 是章节结尾（由调用者保证）
    - 不在对话或动作连续描写中（通过 unsafe 模式排除）
    """
    # 1. 检查是否包含安全关键词
    for pattern in config.safe_break_keywords:
        if re.search(pattern, paragraph):
            return True

    return False


def split_into_paragraphs(text: str, sep: str = "\n\n") -> List[str]:
    """
    将文本按照指定分隔符分割成段落列表

    Args:
        text: 需要分割的文本
        sep: 段落分隔符，默认为双换行符

    Returns:
        List[str]: 段落列表，已去除每个段落的首尾空白
    """
    if not text:
        return []
    # 按分隔符分割文本
    paragraphs = text.split(sep)
    # 去除每个段落的首尾空白，并过滤掉空段落
    cleaned_paragraphs = [para.strip() for para in paragraphs if para.strip()]
    return cleaned_paragraphs


def chunk_novel_text(chapters: List[str]) -> List[TextChunk]:
    """
    将小说章节列表切分为语义连贯的 chunks。

    Args:
        chapters: List[str]，每个元素是一章的完整文本（不含章节标题）

    Returns:
        List[TextChunk]：分块结果列表
    """
    # 分块相关的配置
    config = ChunkConfig(**(get_settings('/home/zhy/workspace/novel_knowledge_base/config/chunk_config.yaml')['chunking']))
    # 当前所有chunks
    chunks: List[TextChunk] = []
    # 当前chunk文本
    current_chunk_text = ""
    current_start_chapter = 1   # 当前 chunk 起始章节，第一章开始
    current_chapter_index = 0   # chunk索引
    llm_client = QwenClient(use_cache=True)

    while current_chapter_index < len(chapters):
        chapter_text = chapters[current_chapter_index]
        chapter_num = current_chapter_index + 1

        # 将当前章节按段落拆分
        paragraphs = split_into_paragraphs(chapter_text, sep=config.paragraph_separator)

        for para in paragraphs:
            # 临时追加当前段落
            temp_text = (current_chunk_text + "\n\n" + para).strip()
            temp_tokens = llm_client.count_tokens(temp_text)

            # 计算追加后未超上限 -> 继续累积
            if temp_tokens <= config.max_tokens_per_chunk:
                current_chunk_text = temp_text
                continue

            # 情况2：当前 chunk 已足够大（>= min），且当前段落是安全断点 -> 可以直接切分
            if (
                    llm_client.count_tokens(current_chunk_text) >= config.min_tokens_per_chunk
                    and is_safe_break_point(para, config)
            ):
                # 切分当前 chunk（不含当前段落）
                chunks.append(_create_chunk(
                    current_chunk_text,
                    current_start_chapter,
                    chapter_num,
                    is_natural_break=True,
                    break_reason="安全关键词或段落结尾",
                    llm_client=llm_client
                ))
                # 重置，当前段落作为新 chunk 开头
                current_chunk_text = para
                current_start_chapter = chapter_num
                break
            elif (llm_client.count_tokens(current_chunk_text) >= config.min_tokens_per_chunk
                  and config.use_llm_for_refinement):
                # 尝试使用 LLM 根据剧情自然断点分段
                chunks_text = _refine_chunks_with_llm(current_chunk_text, llm_client)
                if len(chunks_text) == 0:
                    # 强制切分（不包含当前段落）
                    chunks.append(_create_chunk(
                        current_chunk_text,
                        current_start_chapter,
                        chapter_num,
                        is_natural_break=False,
                        break_reason="达到最大 token 限制，强制切分",
                        llm_client=llm_client
                    ))
                else:
                    chunks.extend([_create_chunk(chunk_text, current_start_chapter, chapter_num, is_natural_break=True, break_reason="大模型判断结果是剧情分段", llm_client=llm_client) for chunk_text in chunks_text])
                # 重置，当前段落作为新 chunk 开头
                current_chunk_text = para
                current_start_chapter = chapter_num

        else:
            # 本章所有段落已处理完，继续下一章
            current_chapter_index += 1
            continue

        # 如果在段落循环中 break（即已切分），继续处理剩余章节
        current_chapter_index += 1

    # 处理最后一个 chunk
    if current_chunk_text.strip():
        chunks.append(_create_chunk(
            current_chunk_text,
            current_start_chapter,
            len(chapters),
            is_natural_break=True,
            break_reason="文本结束",
            llm_client=llm_client
        ))

    return chunks


def _create_chunk(
        text: str,
        start_chapter: int,
        end_chapter: int,
        is_natural_break: bool,
        break_reason: str,
        llm_client: QwenClient = None
) -> TextChunk:
    """辅助函数：创建 TextChunk 对象"""
    return TextChunk(
        start_chapter_idx=start_chapter,
        end_chapter_idx=end_chapter,
        text=text.strip(),
        estimated_tokens=llm_client.count_tokens(text) if llm_client else len(text),
        is_natural_break=is_natural_break,
        break_reason=break_reason
    )


def _refine_chunks_with_llm(
        current_chunk_text: str,
        llm_client: QwenClient
) -> List[str]:
    """
    （预留）使用 LLM 对分块边界进行优化。
    """
    # TODO: 实现 LLM 辅助逻辑
    prompt= """
    你是一个专业的网络小说结构分析师。请分析以下小说文本片段，判断其中是否包含明显的“剧情转场”信号。如果有剧情转场，在转场处划分分段，最多分成两段，在你认为最明显的地方切分。

剧情转场包括但不限于：
- 时间跳跃（如“三年后”“翌日”）
- 地点切换（如“另一边”“此时在东荒”）
- 主线目标变更（如“从今天起，我要……”）
- 新篇章开启（如章节标题含“卷”“篇”“秘境开启”等）
- 重大事件结束后的总结与展望
- 视角切换（如“而在千里之外……”）

请严格按以下 JSON 格式输出：
{
  "has_transition": true/false,
  "transition_type": ["time_jump", "location_shift", "goal_change", "new_arc", "event_conclusion", "perspective_switch", "other"],
  "evidence": "原文句段",
  "chunk_content": ["分段1", "分段2"]
}

has_transition表示是否含有剧情转场信号，是一个bool值，
transition_type表示有哪些类型能说明剧情转场，是一个list，
evidence是原文中支持判断的关键句子（最多2句），如果没有分段就填空串，
chunk_content是分段后的原文：如果有两个分段，务必保证两个分段不重复不遗漏都是原文内容，且组合起来是完整的用户输入文本；如果没有进行分段就返回空列表。
注意不要包含任何额外说明。

输入示例：
三年之后，林风终于破关而出。此刻的他，已踏入元婴之境。与此同时，在北域魔宗，一场针对他的阴谋正在悄然酝酿……

输出示例：
{
  "has_transition": true,
  "transition_type": ["time_jump", "location_shift", "perspective_switch"],
  "evidence": "三年之后，林风终于破关而出。与此同时，在北域魔宗，一场针对他的阴谋正在悄然酝酿……",
  "chunk_content": ["三年之后，林风终于破关而出", "与此同时，在北域魔宗，一场针对他的阴谋正在悄然酝酿……"]
}
    """
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": "请分析以下片段：\n" + current_chunk_text
        }
    ]
    response = llm_client.chat_completion(messages, response_format='json')
    import json
    content = json.loads(response['content'])
    chunks = content['chunk_content']
    return chunks


