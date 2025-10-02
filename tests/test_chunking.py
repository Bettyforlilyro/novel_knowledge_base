# tests/test_chunking.py
import logging
from unittest.mock import Mock, patch

from src.core.text.chunking import (
    ChunkConfig,
    is_safe_break_point,
    split_into_paragraphs,
    chunk_novel_text,
    _create_chunk
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_split_into_paragraphs():
    """测试段落分割功能"""
    print("=== 测试段落分割功能 ===")

    try:
        # 测试基本分割
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        result = split_into_paragraphs(text)
        expected = ["第一段内容。", "第二段内容。", "第三段内容。"]
        assert result == expected, f"基本分割测试失败: 期望 {expected}, 实际 {result}"

        # 测试自定义分隔符
        text = "第一段内容。|||第二段内容。|||第三段内容。"
        result = split_into_paragraphs(text, sep="|||")
        expected = ["第一段内容。", "第二段内容。", "第三段内容。"]
        assert result == expected, f"自定义分隔符测试失败: 期望 {expected}, 实际 {result}"

        # 测试空文本
        result = split_into_paragraphs("")
        assert result == [], f"空文本测试失败: 期望 [], 实际 {result}"

        # 测试单个段落
        text = "只有一个段落的内容。"
        result = split_into_paragraphs(text)
        assert result == ["只有一个段落的内容。"], f"单段落测试失败: 期望 ['只有一个段落的内容。'], 实际 {result}"

        print("段落分割功能测试通过\n")
        return True
    except Exception as e:
        print(f"段落分割功能测试失败: {e}\n")
        return False


def test_is_safe_break_point():
    """测试安全断点判断功能"""
    print("=== 测试安全断点判断功能 ===")

    config = ChunkConfig(
        max_tokens_per_chunk=1000,
        min_tokens_per_chunk=100,
        paragraph_separator="\n\n",
        safe_break_keywords=["数日后", "与此同时"],
        unsafe_break_patterns=[r'“[^”]*$'],
        use_llm_for_refinement=False
    )

    try:
        # 测试包含安全关键词的情况
        paragraph = "数日后，他终于回到了家乡。"
        result = is_safe_break_point(paragraph, config)
        assert result is True, f"安全关键词测试失败: 期望 True, 实际 {result}"

        # 测试匹配不安全模式的情况
        paragraph = "他说道：“我还不能离开这里"
        result = is_safe_break_point(paragraph, config)
        assert result is False, f"不安全模式测试失败: 期望 False, 实际 {result}"

        # 测试以句号结尾的安全情况
        paragraph = "他终于完成了任务。"
        result = is_safe_break_point(paragraph, config)
        assert result is True, f"句号结尾测试失败: 期望 True, 实际 {result}"

        # 测试以逗号结尾的不安全情况
        paragraph = "他还没有完成任务，"
        result = is_safe_break_point(paragraph, config)
        assert result is False, f"逗号结尾测试失败: 期望 False, 实际 {result}"

        print("安全断点判断功能测试通过\n")
        return True
    except Exception as e:
        print(f"安全断点判断功能测试失败: {e}\n")
        return False


def test_create_chunk():
    """测试创建chunk功能"""
    print("=== 测试创建chunk功能 ===")

    try:
        # 测试带LLM的chunk创建
        mock_client = Mock()
        mock_client.count_tokens.return_value = 50

        chunk = _create_chunk(
            text="测试文本内容",
            start_chapter=1,
            end_chapter=2,
            is_natural_break=True,
            break_reason="测试原因",
            llm_client=mock_client
        )

        assert chunk.start_chapter_idx == 1, f"起始章节测试失败: 期望 1, 实际 {chunk.start_chapter_idx}"
        assert chunk.end_chapter_idx == 2, f"结束章节测试失败: 期望 2, 实际 {chunk.end_chapter_idx}"
        assert chunk.text == "测试文本内容", f"文本内容测试失败: 期望 '测试文本内容', 实际 {chunk.text}"
        assert chunk.estimated_tokens == 50, f"Token数测试失败: 期望 50, 实际 {chunk.estimated_tokens}"
        assert chunk.is_natural_break is True, f"自然断点测试失败: 期望 True, 实际 {chunk.is_natural_break}"
        assert chunk.break_reason == "测试原因", f"断点原因测试失败: 期望 '测试原因', 实际 {chunk.break_reason}"

        # 测试不带LLM的chunk创建
        chunk = _create_chunk(
            text="测试文本内容",
            start_chapter=1,
            end_chapter=2,
            is_natural_break=False,
            break_reason="测试原因",
            llm_client=None
        )

        assert chunk.estimated_tokens == len("测试文本内容"), f"无LLM时Token数测试失败"
        assert chunk.is_natural_break is False, f"非自然断点测试失败: 期望 False, 实际 {chunk.is_natural_break}"

        print("创建chunk功能测试通过\n")
        return True
    except Exception as e:
        print(f"创建chunk功能测试失败: {e}\n")
        return False


def test_chunk_novel_text_basic():
    """测试小说文本分块基础功能"""
    print("=== 测试小说文本分块基础功能 ===")

    # 创建测试配置
    test_config = {
        'chunking': {
            'max_tokens_per_chunk': 100,
            'min_tokens_per_chunk': 30,
            'paragraph_separator': '\n\n',
            'safe_break_keywords': ['数日后', '与此同时'],
            'unsafe_break_patterns': [r'“[^”]*$'],
            'use_llm_for_refinement': False
        }
    }

    with patch('src.core.text.chunking.get_settings') as mock_get_settings, \
            patch('src.core.text.chunking.QwenClient') as mock_qwen_client:

        try:
            mock_get_settings.return_value = test_config
            mock_client = Mock()
            mock_client.count_tokens.side_effect = lambda text: len(text) // 2  # 简单模拟token计算
            mock_qwen_client.return_value = mock_client

            # 测试单个小型章节
            chapters = ["这是第一章的第一段。\n\n这是第一章的第二段。"]
            chunks = chunk_novel_text(chapters)

            assert len(chunks) == 1, f"单章分块数量测试失败: 期望 1, 实际 {len(chunks)}"
            assert chunks[0].start_chapter_idx == 1, f"起始章节索引测试失败: 期望 1, 实际 {chunks[0].start_chapter_idx}"
            assert chunks[0].end_chapter_idx == 1, f"结束章节索引测试失败: 期望 1, 实际 {chunks[0].end_chapter_idx}"
            assert "这是第一章的第一段" in chunks[0].text, "文本内容不正确"
            assert chunks[
                       0].break_reason == "文本结束", f"断点原因测试失败: 期望 '文本结束', 实际 {chunks[0].break_reason}"

            print("小说文本分块基础功能测试通过\n")
            return True
        except Exception as e:
            print(f"小说文本分块基础功能测试失败: {e}\n")
            return False


def test_real_novel_chunk():
    """测试小说文本分块功能"""
    print("=== 测试小说文本分块功能 ===")
    novel_path = '/home/zhy/workspace/novel_knowledge_base/data/crawler_novels/历史穿越/呦呦鹿鸣_夕熙.txt'
    chapters = load_novel_text(novel_path)
    chunks = chunk_novel_text(chapters)
    print(chunks)

def load_novel_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return split_into_paragraphs(f.read(), '\n\n')  # 返回划分的章节列表



def test_chunk_novel_text_edge_cases():
    """测试小说文本分块边缘情况"""
    print("=== 测试小说文本分块边缘情况 ===")

    # 创建测试配置
    test_config = {
        'chunking': {
            'max_tokens_per_chunk': 100,
            'min_tokens_per_chunk': 30,
            'paragraph_separator': '\n\n',
            'safe_break_keywords': ['数日后', '与此同时'],
            'unsafe_break_patterns': [r'“[^”]*$'],
            'use_llm_for_refinement': False
        }
    }

    with patch('src.core.text.chunking.get_settings') as mock_get_settings, \
            patch('src.core.text.chunking.QwenClient') as mock_qwen_client:

        try:
            mock_get_settings.return_value = test_config
            mock_client = Mock()
            mock_client.count_tokens.side_effect = lambda text: len(text) // 2
            mock_qwen_client.return_value = mock_client

            # 测试空章节列表
            chapters = []
            chunks = chunk_novel_text(chapters)
            assert len(chunks) == 0, f"空章节列表测试失败: 期望 0, 实际 {len(chunks)}"

            # 测试空章节内容
            chapters = [""]
            chunks = chunk_novel_text(chapters)
            assert len(chunks) == 0, f"空章节内容测试失败: 期望 0, 实际 {len(chunks)}"

            print("小说文本分块边缘情况测试通过\n")
            return True
        except Exception as e:
            print(f"小说文本分块边缘情况测试失败: {e}\n")
            return False


def test_chunk_novel_text_complex():
    """测试复杂的小说文本分块场景"""
    print("=== 测试复杂的小说文本分块场景 ===")

    # 创建测试配置
    test_config = {
        'chunking': {
            'max_tokens_per_chunk': 50,  # 设置较小值以触发强制分割
            'min_tokens_per_chunk': 20,
            'paragraph_separator': '\n\n',
            'safe_break_keywords': ['数日后', '与此同时'],
            'unsafe_break_patterns': [r'“[^”]*$'],
            'use_llm_for_refinement': False
        }
    }

    with patch('src.core.text.chunking.get_settings') as mock_get_settings, \
            patch('src.core.text.chunking.QwenClient') as mock_qwen_client:

        try:
            mock_get_settings.return_value = test_config
            mock_client = Mock()
            mock_client.count_tokens.side_effect = lambda text: len(text) // 2
            mock_qwen_client.return_value = mock_client

            # 测试会触发强制分割的长文本
            chapters = ["很长的第一段内容，内容很多，会超过token限制。" * 5]
            chunks = chunk_novel_text(chapters)

            # 应该被分割成多个chunks
            assert len(chunks) > 1, f"强制分割测试失败: 期望 >1 个chunk, 实际 {len(chunks)}"
            # 检查是否包含强制切分的情况
            forced_break_found = any("强制切分" in chunk.break_reason for chunk in chunks)
            assert forced_break_found, "未找到强制切分的chunk"

            print("复杂的小说文本分块场景测试通过\n")
            return True
        except Exception as e:
            print(f"复杂的小说文本分块场景测试失败: {e}\n")
            return False


def main():
    """主测试函数"""
    print("开始测试 Chunking 模块\n")

    tests = [
        test_split_into_paragraphs,
        test_is_safe_break_point,
        test_create_chunk,
        test_chunk_novel_text_basic,
        test_chunk_novel_text_edge_cases,
        test_chunk_novel_text_complex
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"测试完成: {passed}/{total} 个测试通过")

    if passed == total:
        print("所有测试都通过了！")
    else:
        print("部分测试失败，请检查代码。")


if __name__ == "__main__":
    main()
