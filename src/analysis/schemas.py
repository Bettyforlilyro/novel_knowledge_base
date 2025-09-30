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
    relationship_with_others: Dict[str, str] = Field(
        default_factory=dict,
        description="与其他角色的关系映射，键为对方姓名，值为关系描述（如'师徒'、'死敌'、'暧昧'）"
    )


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
    name: str = Field(..., description="体系名称，如'斗气'、'灵力'、'科技侧基因进化'")
    description: str = Field(..., description="力量体系的规则与表现，如等级划分、核心特征，需随剧情更新")


class Organization(BaseModel):
    """
    组织/势力实体
    """
    name: str = Field(..., description="组织/势力名称")
    description: str = Field(..., description="组织/势力背景与特征，如'道宗'、'魔渊'")
    status: str = Field(..., description="当前状态，如'活跃'、'已覆灭'、'封印中'")
    first_seen_chapter: int = Field(..., description="首次出现章节")
    relationships: Dict[str, str] = Field(
        default_factory=dict,
        description="与其他组织/势力的关系，键为对方名称，值为关系（如'敌对'、'附属'、'同盟'）"
    )


class WorldSettings(BaseModel):
    """
    世界设定
    """
    power_systems: List[PowerSystem] = Field(
        default_factory=list,
        description="书中的所有力量/能力体系"
    )
    organizations: List[Organization] = Field(
        default_factory=list,
        description="书中的所有组织/势力"
    )
    current_map: str = Field(
        ...,
        description="当前剧情所处的地图或场景层级，如'青阳镇→炎城→大荒郡→中州'，反映世界观展开进度"
    )
    event_settings: List[str] = Field(
        default_factory=list,
        description="世界事件设定，比如每千年一次正魔之战、宗门大比、秘境开启等周期性或标志性事件"
    )


# ================
# 2. 工作记忆状态（滚动摘要）
# ================

class WorkingMemoryState(BaseModel):
    """
    LLM 每次分析时携带的“前文摘要”状态
    """
    plot_summary: str = Field(..., description="主线剧情进展，3-5句话")
    characters: List[CharacterState] = Field(
        default_factory=list,
        description="当前已追踪的主要角色状态列表"
    )
    foreshadows: List[ForeshadowRecord] = Field(
        default_factory=list,
        description="所有已发现且未被删除的伏笔记录（含已解决和未解决）"
    )
    world_settings: WorldSettings = Field(..., description="当前世界设定的完整快照")
    current_events_and_goals: str = Field(
        ...,
        description="主角或核心角色当前参与的主要事件、目标与冲突"
    )
    writing_style: str = Field(
        ...,
        description="作者整体写作风格，如'热血爽文'，'苟道流网文'，'杀伐果断'，'稳健发育流'"
    )
    current_stage_id: str = Field(
        default="stage_0",
        description="当前所处剧情阶段ID，如 'volume_1'、'arc_heavenly_trial'"
    )


# ================
# 3. LLM 单次分析输出（结构化响应）
# ================

class ChunkAnalysisResult(BaseModel):
    """
    LLM 对单个 chunk 的分析结果
    """
    updated_plot_summary: str = Field(
        ...,
        description="基于当前 chunk 更新后的主线剧情摘要"
    )
    updated_characters: List[CharacterState] = Field(
        ...,
        description="本 chunk 中出现或状态变更的角色（全量更新，非增量）"
    )
    new_foreshadows: List[ForeshadowRecord] = Field(
        ...,
        description="本 chunk 中新发现的伏笔"
    )
    resolved_foreshadow_ids: List[str] = Field(
        ...,
        description="本 chunk 中被回收的伏笔 ID 列表，如 ['F1', 'F3']"
    )
    updated_world_settings: WorldSettings = Field(
        ...,
        description="更新后的世界设定（全量）"
    )
    current_events_and_goals: str = Field(
        ...,
        description="当前角色目标与事件（覆盖式更新）"
    )
    writing_style: str = Field(
        ...,
        description="本段作者写作风格，如'热血爽文'，'苟道流网文'，'杀伐果断'"
    )
    is_stage_end: bool = Field(
        default=False,
        description="本 chunk 是否标志一个剧情阶段的结束（如卷末）"
    )
    stage_id: Optional[str] = Field(
        default=None,
        description="若阶段结束，给出新的阶段 ID，如 'volume_2'；否则为 None"
    )


# ================
# 4. 阶段归档快照（长期记忆）
# ================

class StageArchiveEntry(BaseModel):
    """
    一个剧情阶段的归档快照（用于回溯）
    """
    stage_id: str = Field(..., description="阶段唯一标识符")
    chapter_range: Dict[str, int] = Field(
        ...,
        description="本阶段覆盖的章节范围，格式如 {'start': 1, 'end': 120}"
    )
    full_plot_summary: str = Field(..., description="本阶段完整剧情摘要")
    characters: List[CharacterState] = Field(
        ...,
        description="阶段结束时所有主要角色的状态快照"
    )
    foreshadows: List[ForeshadowRecord] = Field(
        ...,
        description="阶段结束时所有伏笔的状态（含 resolved 标记）"
    )
    world_entities: WorldSettings = Field(..., description="阶段结束时的世界设定快照")
    key_events: List[Dict[str, Any]] = Field(
        ...,
        description="本阶段关键事件列表，每个元素为 {'chapter': int, 'event': str}"
    )


# ================
# 5. 最终报告（用户输出）
# ================

class FinalReport(BaseModel):
    """
    最终结构化分析报告（将被校验并保存到 processed_report/）
    """
    novel_title: str = Field(..., description="小说标题")
    total_chapters: int = Field(..., description="小说总章节数")
    plot_outline: Dict[str, str] = Field(
        ...,
        description="分阶段剧情大纲，键为阶段ID，值为摘要，如 {'volume_1': '...', 'volume_2': '...'}"
    )
    characters: List[CharacterState] = Field(
        ...,
        description="全书所有主要角色的最终状态"
    )
    unresolved_foreshadows: List[ForeshadowRecord] = Field(
        ...,
        description="未被回收的伏笔列表"
    )
    resolved_foreshadows: List[ForeshadowRecord] = Field(
        ...,
        description="已被回收的伏笔列表"
    )
    world_settings: WorldSettings = Field(..., description="全书最终世界设定")
    stages: List[str] = Field(
        ...,
        description="所有剧情阶段ID列表，按顺序排列，如 ['volume_1', 'volume_2']"
    )