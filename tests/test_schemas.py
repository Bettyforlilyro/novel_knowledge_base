import pytest
from src.analysis.schemas import (
    CharacterState,
    ForeshadowRecord,
    PowerSystem,
    Organization,
    WorldSettings,
    WorkingMemoryState,
    ChunkAnalysisResult,
    StageArchiveEntry,
    FinalReport
)


def test_character_state_initialization():
    """测试角色状态模型"""
    char = CharacterState(
        name="萧炎",
        role="主角",
        style="杀伐果断，重情重义，扮猪吃虎",
        status="斗宗巅峰",
        skills=["焚决", "三千焱炎火", "灵魂感知"],
        first_seen_chapter=1,
        last_updated_chapter=1200,
        relationship_with_others={"药老": "师徒", "纳兰嫣然": "前未婚妻（敌对）"}
    )
    assert char.name == "萧炎"
    assert "焚决" in char.skills
    assert char.relationship_with_others["药老"] == "师徒"
    assert isinstance(char.first_seen_chapter, int)


def test_foreshadow_record():
    """测试伏笔记录"""
    foreshadow = ForeshadowRecord(
        id="F1",
        description="戒指中藏有神秘灵魂体",
        first_seen_chapter=3,
        resolved=True,
        resolved_chapter=50
    )
    assert foreshadow.resolved is True
    assert foreshadow.resolved_chapter == 50

    # 测试未解决伏笔
    foreshadow2 = ForeshadowRecord(
        id="F2",
        description="远古八族的秘密",
        first_seen_chapter=200
    )
    assert foreshadow2.resolved is False
    assert foreshadow2.resolved_chapter is None


def test_world_settings():
    """测试世界设定组合"""
    power = PowerSystem(
        name="斗气",
        description="分九段，从斗之气到斗帝，可凝聚斗气化翼"
    )
    org = Organization(
        name="魂殿",
        description="反派势力，专门抓捕灵魂体",
        status="活跃",
        first_seen_chapter=300,
        relationships={"萧族": "死敌", "丹塔": "敌对"}
    )
    world = WorldSettings(
        power_systems=[power],
        organizations=[org],
        current_map="中州 → 远古遗迹",
        event_settings=["丹会", "天墓开启", "魂天帝复活仪式"]
    )
    assert world.current_map == "中州 → 远古遗迹"
    assert len(world.organizations) == 1
    assert world.organizations[0].relationships["萧族"] == "死敌"


def test_working_memory_state():
    """测试工作记忆状态初始化"""
    wm = WorkingMemoryState(
        plot_summary="萧炎离开加玛帝国，前往中州寻找净莲妖火。",
        world_settings=WorldSettings(
            power_systems=[],
            organizations=[],
            current_map="中州",
            event_settings=[]
        ),
        current_events_and_goals="参加丹会，提升炼药术，打听净莲妖火下落",
        writing_style="热血爽文，升级打脸流"
    )
    assert wm.current_stage_id == "stage_0"  # 默认值
    assert len(wm.characters) == 0  # 默认空列表
    assert wm.writing_style == "热血爽文，升级打脸流"


def test_chunk_analysis_result():
    """测试 LLM 单次分析输出"""
    result = ChunkAnalysisResult(
        updated_plot_summary="萧炎在丹会夺冠，震惊中州。",
        updated_characters=[],
        new_foreshadows=[],
        resolved_foreshadow_ids=["F1"],
        updated_world_settings=WorldSettings(
            power_systems=[],
            organizations=[],
            current_map="中州",
            event_settings=["丹会"]
        ),
        current_events_and_goals="准备进入天墓",
        writing_style="热血爽文",
        is_stage_end=True,
        stage_id="volume_3"
    )
    assert result.is_stage_end is True
    assert result.stage_id == "volume_3"
    assert "F1" in result.resolved_foreshadow_ids


def test_stage_archive_entry():
    """测试阶段归档条目"""
    entry = StageArchiveEntry(
        stage_id="volume_2",
        chapter_range={"start": 301, "end": 600},
        full_plot_summary="萧炎闯荡中州，结识众多强者。",
        characters=[],
        foreshadows=[],
        world_entities=WorldSettings(
            power_systems=[],
            organizations=[],
            current_map="中州",
            event_settings=[]
        ),
        key_events=[
            {"chapter": 400, "event": "丹会开始"},
            {"chapter": 450, "event": "萧炎夺冠"}
        ]
    )
    assert entry.chapter_range["start"] == 301
    assert len(entry.key_events) == 2


def test_final_report():
    """测试最终报告结构"""
    report = FinalReport(
        novel_title="斗破苍穹",
        total_chapters=1648,
        plot_outline={
            "volume_1": "加玛帝国篇",
            "volume_2": "中州篇",
            "volume_3": "大结局篇"
        },
        characters=[],
        unresolved_foreshadows=[],
        resolved_foreshadows=[],
        world_settings=WorldSettings(
            power_systems=[],
            organizations=[],
            current_map="大千世界",
            event_settings=[]
        ),
        stages=["volume_1", "volume_2", "volume_3"]
    )
    assert report.novel_title == "斗破苍穹"
    assert report.total_chapters == 1648
    assert "volume_1" in report.plot_outline


def test_schema_validation_failure():
    """测试 Pydantic 类型校验（应抛出异常）"""
    with pytest.raises(ValueError):
        # 错误：first_seen_chapter 应为 int，不是 str
        CharacterState(
            name="测试角色",
            role="测试",
            style="测试",
            status="测试",
            first_seen_chapter="第一章",  # ❌ 类型错误
            last_updated_chapter=10
        )