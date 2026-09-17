from ctx_weft.protocols.template import LoopConfig


def test_compact_target_ratio_defaults_below_trigger():
    """默认必须构成**真的滞后区**（target < trigger），不是曾经的 0.0（= 无滞后）。

    无滞后时 compact 把估算压到「刚低于 compact_token_ratio」就停，ActStep 续跑第一轮加上
    本轮输出立刻又越过停机线，一次上下文恢复配额当场蒸发（见 act._can_recover_context）。
    这条断言守的是「有滞后」这个性质，故不写死 0.4——调默认值不该让它失败。
    """
    lc = LoopConfig()
    assert 0.0 < lc.compact_target_ratio < lc.compact_token_ratio


def test_compact_target_ratio_below_trigger():
    lc = LoopConfig(compact_token_ratio=0.8, compact_target_ratio=0.6)
    assert lc.compact_target_ratio < lc.compact_token_ratio


def test_stop_ratio_above_compact_trigger():
    """act 的停机线必须高于 compact 的触发线（两条线分工，不重合）。

    重合时 prepare 刚压到「刚低于触发线」，act 第一轮就又越停机线——恢复机制空转。
    """
    lc = LoopConfig()
    assert lc.context_limit_stop_ratio > lc.compact_token_ratio
