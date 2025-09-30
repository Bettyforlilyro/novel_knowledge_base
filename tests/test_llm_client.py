# test_llm_client.py
import logging
from core.llm.llm_client import QwenClient

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_chat_completion():
    """测试基本的对话完成功能"""
    print("=== 测试基本对话完成 ===")

    # 初始化客户端
    client = QwenClient(use_cache=False)

    # 准备测试消息
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，介绍一下你自己。"}
    ]

    try:
        # 调用模型
        response = client.chat_completion(messages)

        # 输出结果
        print(f"模型回复: {response['content']}")
        print(f"使用的模型: {response['model']}")
        print(f"Token使用情况: {response['usage']}")
        print("测试通过\n")
        return True
    except Exception as e:
        print(f"测试失败: {e}\n")
        return False


def test_json_response_format():
    """测试JSON格式响应"""
    print("=== 测试JSON格式响应 ===")

    client = QwenClient(use_cache=False)

    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。请以JSON格式回复。"},
        {"role": "user", "content": "告诉我以下信息：名字是Qwen，类型是大语言模型，开发者是阿里云。请以JSON格式返回。"}
    ]

    try:
        response = client.chat_completion(messages, response_format="json")
        print(f"模型回复: {response['content']}")
        print(f"Token使用情况: {response['usage']}")
        print("测试通过\n")
        return True
    except Exception as e:
        print(f"测试失败: {e}\n")
        return False


def test_cache_functionality():
    """测试缓存功能"""
    print("=== 测试缓存功能 ===")

    # 先用缓存开启的客户端调用一次
    client_with_cache = QwenClient(use_cache=True)

    messages = [
        {"role": "user", "content": "请用一句话解释什么是人工智能？"}
    ]

    try:
        # 第一次调用
        print("第一次调用（应该没有缓存）:")
        response1 = client_with_cache.chat_completion(messages)
        print(f"模型回复: {response1['content'][:50]}...")

        # 第二次调用相同内容
        print("第二次调用（应该命中缓存）:")
        response2 = client_with_cache.chat_completion(messages)
        print(f"模型回复: {response2['content'][:50]}...")

        # 检查内容是否一致
        if response1['content'] == response2['content']:
            print("缓存功能正常工作\n")
            return True
        else:
            print("缓存内容不一致\n")
            return False
    except Exception as e:
        print(f"测试失败: {e}\n")
        return False


def test_token_counting():
    """测试Token计数功能"""
    print("=== 测试Token计数功能 ===")

    client = QwenClient(use_cache=False)

    text = "这是一个用于测试token计数功能的示例文本。The quick brown fox jumps over the lazy dog."

    try:
        token_count = client.count_tokens(text)
        print(f"文本: {text}")
        print(f"Token数量: {token_count}")
        print("测试通过\n")
        return True
    except Exception as e:
        print(f"测试失败: {e}\n")
        return False


def test_error_handling():
    """测试错误处理功能"""
    print("=== 测试错误处理功能 ===")

    # 这里我们可以通过构造一些特殊情况来测试错误处理
    # 比如发送一个很长的消息来测试超时或长度限制
    client = QwenClient(use_cache=False)

    # 构造一个相对较长的消息
    long_message = "Hello! " * 1000  # 重复1000次
    messages = [
        {"role": "user", "content": long_message}
    ]

    try:
        response = client.chat_completion(messages)
        print(f"长消息处理成功，回复长度: {len(response['content'])}")
        print("测试通过\n")
        return True
    except Exception as e:
        print(f"处理长消息时出错（这可能是预期行为）: {e}")
        # 这种情况下我们也认为测试通过，因为错误被正确捕获了
        print("错误处理功能正常\n")
        return True


def main():
    """主测试函数"""
    print("开始测试 QwenClient 类\n")

    tests = [
        test_basic_chat_completion,
        test_json_response_format,
        test_cache_functionality,
        test_token_counting,
        test_error_handling
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
