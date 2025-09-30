from jinja2 import Template
import yaml

def render_character_prompt(novel_text):
    with open("prompts/characters.yaml") as f:
        tmpl = yaml.safe_load(f)
    prompt = f"""
    {tmpl['role']}。{tmpl['task']}。

    要求：
    - {tmpl['constraints']}
    - 输出字段：{', '.join(tmpl['fields'])}
    - {tmpl['output_format']}

    小说内容：
    {novel_text[:10000]}  # 可动态截断
    """
    return [{"role": "user", "content": prompt}]