import yaml
from typing import Dict, Any
import os


def get_settings(config_path: str) -> Dict[str, Any]:
    """
    加载指定路径的 YAML 配置文件并返回配置数据

    Args:
        config_path (str): YAML 配置文件的路径

    Returns:
        Dict[str, Any]: 解析后的配置数据字典

    Raises:
        FileNotFoundError: 当配置文件不存在时抛出
        yaml.YAMLError: 当YAML文件格式错误时抛出
    """
    # 检查文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    # 读取并解析 YAML 文件
    with open(config_path, 'r', encoding='utf-8') as file:
        try:
            config_data = yaml.safe_load(file)
            return config_data or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"解析 YAML 文件失败 {config_path}: {e}")

