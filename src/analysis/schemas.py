from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ================
# 1. 核心实体定义
# ================

class CharacterState(BaseModel):
    """
    角色当前状态（仅限主要角色）
    """
    name: str = Field(..., description="角色姓名")
    role: str = Field(..., description="身份/阵营描述，如'主角'、'反派'、'盟友'")
    style: str = Field(..., description="角色人设，性格是杀伐果断、扮猪吃虎，还是冷静智谋、吐槽役？是否讨喜？")
    status: str = Field(..., description="当前状态，如'斗尊修为'、'重伤昏迷'、'已死亡'")
    skills: List[str] = Field(default_factory=list, description="角色技能或者当前能力，包括金手指")
    first_seen_chapter: int = Field(..., description="首次出现章节")
    last_updated_chapter: int = Field(..., description="最后更新章节")
    relationship_with_others: Dict[str, str] = Field(default_factory=dict, description="与其他角色关系")


class ForeshadowRecord(BaseModel):
    """
    伏笔记录（跨章节追踪）
    """
    id: str = Field(..., description="伏笔唯一ID，如 F1, F2")
    description: str = Field(..., description="伏笔内容简述")
    first_seen_chapter: int = Field(..., description="首次出现章节")
    resolved: bool = Field(default=False, description="是否已回收")
    resolved_chapter: Optional[int] = Field(default=None, description="回收章节（若已解决）")


class PowerSystem(BaseModel):
    """
    力量/能力体系
    """
    name: str = Field(..., description="体系名称")
    description: str = Field(..., description="力量描述，随时更新")


class Organization(BaseModel):
    """
    组织/势力实体
    """
    name: str = Field(..., description="组织/势力名称")
    description: str = Field(..., description="组织/势力描述，如'道宗'、'魔渊'")
    status: str = Field(..., description="当前状态，如'活跃'、'已覆灭'、'封印中'")
    first_seen_chapter: int = Field(..., description="首次出现章节")
    relationships: Dict[str, str] = Field(default_factory=dict, description="与其他组织/势力关系")


class WorldSettings(BaseModel):
    """
    世界设定
    """
    power_systems: List[PowerSystem] = Field(default_factory=list, description='书中的所有力量/能力体系')
    organizations: List[Organization] = Field(default_factory=list, description='书中的所有组织/势力')
    current_map: str = Field(..., description="世界观，当前地图或者场景状态")
    event_settings: List[str] = Field(default_factory=list, description="世界事件设定，比如每千年一次正魔之战")


# ================
# 2. 工作记忆状态（滚动摘要）
# ================

class WorkingMemoryState(BaseModel):
    """
    LLM 每次分析时携带的“前文摘要”状态
    """
    plot_summary: str = Field(..., description="主线剧情进展，3-5句话")
    characters: List[CharacterState] = Field(default_factory=list)
    foreshadows: List[ForeshadowRecord] = Field(default_factory=list)
    world_settings: WorldSettings = Field(..., description="世界设定")
    current_events_and_goals: str = Field(..., description="角色当前目标与主要参与事件或者冲突")
    writing_style: str = Field(..., description="作者写作风格，如'热血爽文'，'苟道流网文'，'杀伐果断'")
    current_stage_id: str = Field(default="stage_0", description="当前所处阶段ID，如 volume_1")


# ================
# 3. LLM 单次分析输出（结构化响应）
# ================

class ChunkAnalysisResult(BaseModel):
    """
    LLM 对单个 chunk 的分析结果
    """
    updated_plot_summary: str
    updated_characters: List[CharacterState]
    new_foreshadows: List[ForeshadowRecord]
    resolved_foreshadow_ids: List[str]  # 如 ["F1", "F3"]
    updated_world_settings: WorldSettings
    current_events_and_goals: str
    writing_style: str = Field(..., description="本段作者写作风格，如'热血爽文'，'苟道流网文'，'杀伐果断'")
    is_stage_end: bool = Field(default=False, description="本 chunk 是否标志阶段结束")
    stage_id: Optional[str] = Field(default=None, description="若阶段结束，给出新 stage_id，如 'volume_2'")


# ================
# 4. 阶段归档快照（长期记忆）
# ================

class StageArchiveEntry(BaseModel):
    """
    一个剧情阶段的归档快照（用于回溯）
    """
    stage_id: str
    chapter_range: Dict[str, int]  # e.g., {"start": 1, "end": 120}
    full_plot_summary: str
    characters: List[CharacterState]
    foreshadows: List[ForeshadowRecord]
    world_entities: WorldSettings
    key_events: List[Dict[str, Any]]  # [{"chapter": 50, "event": "宗门大比"}]


# ================
# 5. 最终报告（用户输出）
# ================

class FinalReport(BaseModel):
    """
    最终结构化分析报告（将被校验并保存到 processed_report/）
    """
    novel_title: str
    total_chapters: int
    plot_outline: Dict[str, str]  # e.g., {"act1": "...", "act2": "..."}
    characters: List[CharacterState]
    unresolved_foreshadows: List[ForeshadowRecord]
    resolved_foreshadows: List[ForeshadowRecord]
    world_settings: List[WorldSettings]
    stages: List[str]  # e.g., ["volume_1", "volume_2"]