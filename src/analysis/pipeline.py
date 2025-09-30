from analysis.prompts.render import render_character_prompt
# src/analysis/pipeline.py
from core.llm.llm_client import QwenClient
# from core.chunking import smart_chunk
import json

class NovelAnalyzer:
    def __init__(self):
        self.llm = QwenClient()

    # def analyze(self, novel_text: str, novel_id: str) -> dict:
        # Step 1: 分块（如果太长）
        # chunks = smart_chunk(novel_text, max_tokens=24000)  # 留出输出空间

        # if len(chunks) == 1:
            # 直接全局分析
            # report = self._full_analysis(chunks[0])
        # else:
            # Map-Reduce
            # partial_results = []
            # for i, chunk in enumerate(chunks):
            #     partial = self._chunk_analysis(chunk, chunk_id=i)
            #     partial_results.append(partial)
            # report = self._merge_results(partial_results)

        # report["novel_id"] = novel_id
        # return report

    def _full_analysis(self, text):
        # 并行调用多个维度的分析（可选）
        plot = self._extract_plot(text)
        characters = self._extract_characters(text)
        style = self._extract_style(text)
        return {
            "plot_summary": plot,
            "characters": characters,
            "writing_style": style
        }

    def _extract_characters(self, text):
        messages = render_character_prompt(text)
        resp = self.llm.chat_completion(messages, response_format="json")
        return json.loads(resp.choices[0].message.content)