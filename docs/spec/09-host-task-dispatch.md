# 09 · 宿主侧任务派发：`dispatch_task`

> 给 host 一个与 `delegate_task` 对等的派发入口：在**已有 session** 上开一条**全新 task**，
> 用 `NormalTaskSettings` 表达调度意图，可挂已有 agent、可从已有 agent 派生子 agent、
> 也可以起一棵**与 root 无血缘的全新 agent 树**。
>
> 契约以本文为准。本文只描述新增入口与它要求 core 做的改动，既有路径
> （`start_session` / `send_message` / `delegate_task`）的语义一律不变。

---

## 1. 为什么需要它

今天 host 想「在已有会话上开一条新 task」只有两条路，都不够用：

| 现有入口 | 能开新 task | 能带 `TaskSettings` | 能建新 agent | 会话有在跑任务时可用 |
|---|---|---|---|---|
| `send_message(agent_id, ...)` | 仅当目标 agent 的 `current_task` 已终态，否则**注入现有 task** | ❌ 硬编码默认 `NormalTaskSettings()` | ❌ | ✅ |
| `start_session(session_id=…, resume=True)` | ✅ 新 root task | ✅ `initial_task` | ❌（见下） | ❌ `UnfinishedTasksError` |

第三列两个 ❌ 是同一个原因。`_make_root_task_manager`（`session_registry.py:331`）
建 root task 时**预填** `assigned_agent_id=session.root_agent_id`，而
`_SessionTaskRunner.assemble` 的分支判据正是「`assigned_agent_id` 空不空」——非空走
`materialize`（按同一 id 重新水合），不走 `instantiate`。实测：

```
start_session(initial_task={"use_subagent": True, "subagent_template": "agent:tpl_echo"})
→ AgentInstantiated: [(agt_01M2M3PJ…, 'tpl_echo')]   # 只有 root 一个
→ task assigned to  : agt_01M2M3PJ…                   # 就是 root
```

即 **`use_subagent=True` 在 root task 上是哑弹**。当前全仓唯一能真正 spawn 新 agent 的
是 LLM 调 `delegate_task`——它建的 child 不填 `assigned_agent_id`，于是走到
`instantiate(parent_agent_id=t.creator_agent_id)`。host 没有对等入口。

好消息是缺口很窄，底层机器已经齐了：

- `AgentLifecycleManager.instantiate`（`agent_manager.py:586`）的 `parent_agent_id=None`
  分支已经是**完整的无父路径**：`spawn_depth=0`、不发 `AgentSpawned`、不写 `_children`、
  `llm` 落到 `ModelChoice()`（账号默认）。**一个 session 下多棵 agent 树在结构上已经成立**，
  只是没人从这条路进来过。
- `effective_agent_id`（`runner.py:59`）对 `use_subagent=True` 且 `assigned_agent_id` 为空的
  task 返回 `__sub__{task.id}` 作串行键——不会被误并进 root agent 那一桶。
- `_copy_memory_for_inherit`（`runtime.py:149`）实际按 **agent_id** 取记忆
  （`load_view(MemoryAddress(session_id, agent_id=…))`），`parent_task` 参数只用来推导那个
  agent id 和打一个元数据标签。「继承指定 agent 的记忆」是改签名的事，不是新机制。

---

## 2. 入口

命名用 `dispatch_task`，不用 `start_task`：后者已经是记忆层派发框的叙事名
（`finalize.py:64` `START_TASK_NAME = qualify("control:start_task")`），占用会造成歧义。

### 2.1 签名

```python
async def dispatch_task(
    self,
    session_id: str,
    content: "str | list[ContentPart]",
    *,
    agent_id: str | None = None,
    settings: "NormalTaskSettings | dict | None" = None,
    title: str = "",
    description: str = "",
    interaction_mode: "TaskInteractionMode" = "auto",
    unattended: bool = False,
    llm_account: str | None = None,
    llm_model: str | None = None,
) -> TurnHandle: ...
```

形状对齐 `send_message(agent_id, content, *, …)`：两个位置参数（落点 + 内容），
其余全部 keyword-only。不引入 `DispatchTaskParams` 数据类——`SessionStartParams`
存在的理由是 `start_session` 要把一整包参数跨 create/resume 两条路透传并参与序列化，
本入口没有这个需求，九个参数直接摊开更省一层间接。

### 2.2 逐参数语义

**`session_id`** —— 必填。派发的落点，必须是事件日志里存在的 session。这条 task 是该
session 的**顶层 task**：`parent_task_id=None`，与 root task 平级，不参与
`_try_resume_parent` 的父子唤醒，也不会有任何「父任务等它」的挂起。

**`content`** —— 这条 task 的 `user_prompt`。走与另外三个入口共用的
`_validate_and_normalize_content`（validate → event 侧外部化 → memory 侧外部化）。
支持 `list[ContentPart]` 多模态。

**`agent_id`** —— **血缘与执行的锚点**，语义见 §3 的矩阵：

- 给了 + `use_subagent=False` → 这个 agent 执行这条 task
- 给了 + `use_subagent=True` → 从这个 agent **派生**一个新 agent 来执行
- 不给（必须 `use_subagent=True`）→ 起一棵与 root 无血缘的**全新 agent 树**

**`settings`** —— 与 `delegate_task` 同一个 `NormalTaskSettings`，传 dict 则走
`deserialize_settings`（与 `SessionStartParams.create(initial_task=dict)` 同一口径，
host 从 JSON 配置直接喂）。字段分工：

| 字段 | 在本入口的作用 |
|---|---|
| `use_subagent` | **矩阵的第二根轴**：要不要为这条 task 新建 agent |
| `subagent_template` | 新 agent 的模板，**`use_subagent=True` 时必填**（规范形式 `provider:name`，见 §7.1 V5） |
| `inherit_memory` | 新 agent 起始记忆是否从源 agent 复制。默认 `True` |
| `inherit_from_agent_id` | **本轮新增**：显式指定继承源，见 §6 |
| `skill_name` | 原样透传给该 task |
| `purpose` | 原样透传（默认 `"act"`） |
| `spawn_titles` | **调用方不要设**：瞬态累加器，由 `delegate_*` 写、`SuspendStep` 清。入口一律重置为空 |

**`title` / `description`** —— 与 `delegate_task` 对齐。`title` 非空会让
`act_guidance._task_label` 走 title 分支（否则回落 `user_prompt`），也决定任务交接时
「标题 + id」的称呼（spec: task-handoff）。`description` 留空则取
`content_to_text(normalized)[:200]`。

**`interaction_mode`** —— 默认 **`"auto"`**（与 `Task` 的字段默认一致）。派发出去的是
一件**作业**不是一轮对话，actor 必须调 `finish_task` 收尾。传 `"interactive"` 表示
「我准备好之后用 `send_message` 跟它继续对话」，此时 actor 的纯文本回合会冷 park 等人。
与 `_start_task_for_agent` 的默认相反（那里是用户消息，默认 `interactive`），刻意如此。

**`unattended`** —— 「没有人看顾」，唯一用途是在 `HitlService.open()` 一处堵死 HITL
（`Task.unattended`）。不变式 `unattended ⟹ interaction_mode == "auto"` 在**入口即拒**
（见 §7 V2），**不静默降级**——`_child_mode` 对 LLM 的请求是降级 + warning，那是因为
调用方是模型；host 显式写了两个互斥的参数，响亮报错才对。

**`llm_account` / `llm_model`** —— 只在**新建 agent** 时生效（写进 agent record 的
`ModelChoice`）。省略时沿用 `instantiate` 既有语义：派生 → 继承父 agent 的选择，
新树 → `ModelChoice()` 跟随账号默认。`use_subagent=False` 时传了会被忽略——
既有 agent 的模型选择归 `set_agent_llm`，派发不是改模型的入口。

### 2.3 返回

`TurnHandle`，四个身份字段恒非空。`handle.agent_id` 是**真正执行这条 task 的 agent**
（新建的情形下就是入口刚铸出来的那个 id，host 拿到即可用于后续 `send_message` /
`get_agent` / `cancel_agent` 寻址）。要同步语义就 `await handle.wait_for_finish()`。

### 2.4 派生**不**污染 A 的对话

`use_subagent=True, agent_id=A` 时，A 是**血缘上的父**，不是**对话上的父**。
因为这条 task 的 `parent_task_id` 是 `None`，`finalize` 的两处准入判据
（`finalize.py:257` 的 `if not task.parent_task_id: return`、`:528` 的
`if task.parent_task_id and mem_content`）都不成立：

- 不会往 A 的 agent scope 铸派发框，也不会回写 dispatch result
- 子树自己按 `is_own_root`（`finalize.py:518`）在**自己的** agent scope 合成 finish 对

也就是说，A 从它自己的视角看不到这次派发发生过。这是有意的——host 的派发不该
凭空往一个正在对话的 agent 的上下文里插入它没做过的工具调用。要让 A 知道，
用 `inherit_memory` 的反向（跑完后 host 自己 `send_message` 告诉它）。

---

## 3. 语义矩阵

两根轴：`agent_id` 给不给、`settings.use_subagent` 真不真。入口把它们翻译成 Task 上
两个字段——**这两个字段才是真正驱动一切的东西**：

| `agent_id` | `use_subagent` | `creator_agent_id` | `assigned_agent_id` | 效果 |
|---|---|---|---|---|
| `A` | `False` | `A` | `A` | 挂 A 上跑，延续 A 的对话与记忆 scope |
| `A` | `True` | `A` | 新铸 id | 从 A **派生**子 agent，`spawn_depth = A.depth + 1` |
| `None` | `True` | `""` | 新铸 id | **全新 agent 树**，`parent=None`、`spawn_depth=0` |
| `None` | `False` | — | — | `ValueError`（见下） |

第四格拒绝而不是默认挂 root：「不指定 agent」与「要 root 跑」是两件不同的意图，
默认值替调用方猜哪一件都会在某天变成事故。要 root 就显式
`agent_id=session.root_agent_id`。

**root agent 不变。** 新树的 agent 只是 `AgentLifecycleManager` 里一条
`parent_agent_id=None, spawn_depth=0` 的 record，`Session.root_agent_id` 一个字节都不改。
`agent_ids_of_session` / `cancel_session` / `forget_session` 按 session 收全部成员，
森林与单树对它们没有区别。

### 3.1 为什么重放折得出同一棵森林

`_rebuild_agents`（`reducers.py:426`）从 task 推算树形字段：

```python
use_subagent = task.settings_raw.get("use_subagent", False)
creator = task.creator_agent_id
depth = view.agents[creator].spawn_depth + 1 if (use_subagent and creator in view.agents) else 0
av.parent_agent_id = creator or None
```

于是**不变式**是：`instantiate(parent_agent_id=X)` 里的 `X` 必须恒等于
`task.creator_agent_id or None`。矩阵的三行都满足：

- 派生行：`creator=A` → 内存 `parent=A, depth=A+1`；重放 `parent=A, depth=A+1` ✅
- 新树行：`creator=""` → 内存 `parent=None, depth=0`；重放 `parent=None, depth=0` ✅
- 非 subagent 行：`use_subagent=False` → 重放 `depth=0, parent=creator`，而该 agent 的
  真身早由它自己那次 `AgentInstantiated` 钉住，这里不覆盖 `template_id` ✅

**这条不变式是本设计的地基**，任何改动都不许破坏它。

---

## 4. 执行时序

入口内部严格按这个顺序，**每一步的位置都有理由**：

```
1. 解 tenant_id            ← _tenant_for_session(session_id)，绝不抛
2. 规范化 settings          ← dict → deserialize_settings；校验矩阵第四格 → ValueError
3. 取活 owner TM            ← _task_managers[session_id]
   缺失/已死 → 冷复活：持 _resume_locks[session_id]，_recover_session_locked(keep_alive=True)
              （session 在事件日志里不存在 → RuntimeError 从这里抛出）
4. agent 守卫               ← agent_id 给了则按 §7.2 分岔：
                              use_subagent=False → assert_can_receive(A)   （running 拒）
                              use_subagent=True  → assert_can_parent(A)    （running 放行）
                              顺带校验 A 确属该 session（判据同 send_message）
5. 内容门控                 ← _validate_and_normalize_content(content, session_id, tenant_id)
                              （validate → event 侧外部化 → memory 侧外部化，三入口共用）
6. 铸 task_id               ← generate_id("tsk")
7. 需要新 agent 则 instantiate(
       template_id = settings.subagent_template or <session 模板>,
       parent_agent_id = agent_id or None,      ← §3.1 的不变式
       task_id = <第 6 步的 id>,                ← AgentSpawned/SpawnRejected 的信封要它
       llm = ModelChoice(llm_account, llm_model) if 显式给了 else None,
   )                        ← SpawnDepthExceeded 在这里同步抛出
8. 建 Task（字段见 §4.1）
9. tm.push_task(task, user_prompt_event_jsonable=…, provisional=False)
10. reg.set_current_task(<执行 agent>, task.id)
11. asyncio.create_task(tm.drain())
12. return TurnHandle(...)
```

**第 3 步在第 5 步之前**：拿不到 TM 就什么都不该发生，这一步只读不写。
**第 5 步在第 7 步之前**：格式/blob 门控必须先于任何持久化——第 7 步会发
`AgentInstantiated`，那是这条路径上第一个落盘的东西（与 `run_single_task` /
`start_session` 同一条纪律）。
**第 7 步在第 8 步之前**：`TurnHandle.agent_id` 恒非空要求入口返回时就知道执行 agent 是谁；
让 `assemble` 去 instantiate 就来不及了。预建之后 `assemble` 走 `materialize` 分支
（`assigned_agent_id` 已非空），与「子 agent 重派发 / 恢复」走的是同一条路，不新增分支。

> **一个诚实的例外**：第 7 步成功、第 9 步失败（存储隔离等），会在事件流里留下一个
> 无 task 可跑的孤儿 agent record + `AgentInstantiated`。不为此加补偿删除——
> `AgentLifecycleManager` 没有删除语义，且这与 `_validate_and_normalize_content`
> 已记录的「event 侧孤儿字节」是同一类、同量级的取舍。孤儿 agent 是 `idle` 的，
> 不占调度、不影响 `is_done()`。

### 4.1 Task 字段

```python
Task(
    id=<第 6 步>,
    session_id=session_id,
    status="ACTIVE",
    tenant_id=<第 1 步>,
    parent_task_id=None,                    # 顶层 task，不参与父子唤醒
    creator_agent_id=agent_id or "",        # §3.1 的不变式
    assigned_agent_id=<新 agent id 或 agent_id>,
    title=title,
    description=description or content_to_text(normalized)[:200],
    user_prompt=normalized,
    user_prompt_event_jsonable=<第 5 步的第二个产物>,
    settings=settings,
    unattended=unattended,
    interaction_mode="auto" if unattended else "interactive",
    created_at=now_utc(),
)
```

`provisional=False`（对比 `_start_task_for_agent` 的 `True`）：未提交窗口是给
「一条用户消息开出的一轮」准备的——LLM 没开口前不算发生、用户按暂停可整轮丢弃。
host 的一次显式派发不是对话轮，**创建即事实**，与 `_flush_staged` 推委派子任务同口径。

---

## 5. 事件流

新树（`agent_id=None, use_subagent=True`）：

```
AgentInstantiated{template_id, template_version, llm_account, llm_model}   ← agent_id=新id, task_id=新task
TaskCreated{...}                                                           ← push_task
TaskStarted / RunStarted / ...                                             ← drain 派发
```

派生（`agent_id=A, use_subagent=True`）在 `AgentInstantiated` 之前多一条
`AgentSpawned{parent_agent_id: A, subtask_id: <task>}`；depth 超限时改发
`SpawnRejected{reason: "depth_limit", attempted_subtask_id: <task>}` 并抛
`SpawnDepthExceeded`。挂已有 agent（`use_subagent=False`）不发任何 agent 事件。

**不发 `SessionResumed`**——这不是新一轮会话，是往活着的会话里加一条 task。
因此也**没有 `UnfinishedTasksError` 那道「弃轮禁止」门**：会话里有别的任务在跑时
照样可以派发，这正是相对 `start_session(resume=True)` 的核心差别。

会话成员登记走既有路径：`SessionRegistry._MEMBER_EVENTS`（`session_registry.py:104`）
订阅 `AgentInstantiated` / `AgentSpawned`，`provisional=True` 收未提交窗口内的事件。
**前置条件**：`_states[session_id]` 必须已存在，否则 handler 直接 return、成员集合漏登记。
第 3 步的冷复活路径内部会 `register_session`；热路径下 session 本就在册。**第 3 步必须
先于第 7 步**的第二个理由就是这个。

---

## 6. 记忆继承

`settings.inherit_memory=True` 时，`assemble` 会把**源 agent** 当前的召回视图整份复制进
新 agent 的 scope（`_copy_memory_for_inherit`）。源的选择规则改成三级，**显式优先**：

```
settings.inherit_from_agent_id   （新增字段，host 显式指定）
  ↓ 空则
父任务的 agent                    （parent_task_id → task.assigned or creator）  ← 现有行为
  ↓ 无父任务则
_latest_prior_root_task 的 agent  （上一条 root task）                          ← 现有行为
```

后两级逐字保留，存量行为零变化。

**只在 `use_subagent=True` 时生效。** `use_subagent=False` 的 task 本就跑在目标 agent
自己的 scope 上、手里已经是它的全部记忆，复制无意义——这也是现状（`inherit_memory`
只在 `assemble` 的 subagent 分支被消费），文档写明而非改代码。

**跨树继承是合法的**：新树 agent 与 root 无血缘，但可以 `inherit_from_agent_id=<root>`
把 root 的对话整份带过去。血缘（`parent_agent_id`）与记忆来源（`inherit_from`）在本设计
里是**两个正交的轴**——这正是「不从任何已有 agent 派生，但要它的上下文」这个用例需要的。

---

## 7. 入口校验与错误分类

### 7.1 纯参数校验（第 2 步，零副作用、零 IO）

| # | 条件 | 异常 |
|---|---|---|
| V1 | `agent_id is None` 且 `not settings.use_subagent` | `ValueError` |
| V2 | `unattended` 且 `interaction_mode == "interactive"` | `ValueError` |
| V3 | `settings.inherit_from_agent_id` 非空 且 `not settings.use_subagent` | `ValueError` |
| V4 | `settings` 不是 `NormalTaskSettings`（如喂了 `CompactTaskSettings` 的 dict） | `ValueError` |
| V5 | `use_subagent` 为真 且 `settings.subagent_template` 为空 | `ValueError` |

V1 见 §3：「不指定 agent」与「要 root 跑」是两个意图，不替调用方猜。
V2 见 §2.2：不变式冲突对 host 响亮报错，不学 `_child_mode` 的静默降级。
V5 是**实现期发现的**：本入口自己 `instantiate`（§4 第 7 步），因此必须手握**规范形式**
的模板 id。会话的那一份只活在 `_SessionTaskRunner._template_id` 里——`Session` 不带
`template_id` 字段，agent record 里存的是 `template.id`（裸名），而 `resolve_qualified`
只做 `agent__x → agent:x` 的反规范化、变不出裸名的 provider 前缀。没有可靠的默认可回落，
与其猜一个，不如要求调用方说清楚要造一个什么型号的 agent。（`assemble` 里那条
「留空回落 session 模板」的既有行为原样保留，只是本入口够不到它。）

V3 是**显式旋钮才报错**：`inherit_memory` 默认就是 `True`，`use_subagent=False` 时它
静默无效是既有行为（§6），不改；但 `inherit_from_agent_id` 没有默认值，host 写了它
就一定是有意图的，无效必须说出来。

### 7.2 两道 agent 守卫（第 4 步）

**判据按「这个 agent 要不要亲自执行」分岔**，两条都收敛在
`AgentLifecycleManager` 里，不在 runtime 内联 status 判断（沿用
`assert_can_receive` docstring 立下的规矩：判断逻辑收敛在一处）：

| `use_subagent` | A 的角色 | 守卫 | 放行的状态 |
|---|---|---|---|
| `False` | **执行者** | `assert_can_receive(A)`（既有，`agent_manager.py:378`） | `idle` / `waiting_human` / `interrupted` |
| `True` | **血缘父**，不执行 | `assert_can_parent(A)`（**本轮新增**） | 上述 + `running` |

`assert_can_parent` 只拒两种：不存在（`AgentNotFound`）、`terminated`
（`AgentTerminatedError`）。**`running` 放行**——A 只提供血缘（`parent_agent_id` →
`spawn_depth`）与模型选择的继承，不占用它的执行槽，也不碰它的对话记忆（§2.4）。
这正是 `delegate_task` 一直在做的事：A 自己在跑的时候派生子 agent 是常态；host 从外部
做同一件事没有理由更严。

`terminated` 仍然拒：往一个已被显式 cancel 的 agent 上接子树，接出来的东西归属一个
死掉的父，`cancel_session` 的逐 agent 终态化也会立刻把它一并带走——让它建出来只是
制造垃圾。

**两条守卫都不排队**：拒了就是拒了，调用方自行重试或先 pause/cancel。

### 7.3 其余错误

| 条件 | 异常 | 时点 |
|---|---|---|
| session 不在事件日志里 | `SessionNotFound` / `RuntimeError` | 第 3 步，零副作用 |
| `agent_id` 不属于 `session_id` | `ValueError` | 第 4 步，零副作用（判据同 `send_message`） |
| 内容非法 / 携图但 event blob store 不可外部化 | `InvalidContentError` | 第 5 步，零副作用 |
| `subagent_template` 解不出 | `TemplateNotFoundError` | 第 7 步，**先于任何 emit** |
| 派生深度超限 | `SpawnDepthExceeded`（伴 `SpawnRejected` 事件） | 第 7 步 |

第 7 步之前的每一条都**零副作用**：不发事件、不建 agent、不入队。

---

## 8. Core 改动清单

| # | 位置 | 改动 | 必需性 |
|---|---|---|---|
| C1 | `core/runtime.py` 新增 `dispatch_task` | 入口本体，§4 的 12 步 | 必需 |
| C2 | `core/models/task.py:27` `NormalTaskSettings` | 新增 `inherit_from_agent_id: str = ""` | 必需 |
| C3 | `core/runtime.py:149` `_copy_memory_for_inherit` | 签名从 `parent_task: Task` 改为 `source_agent_id: str` + `source_task_id: str \| None`，并**全部 keyword-only**；元数据补 `inherited_from_agent_id`，`inherited_from_task_id` 在有源 task 时照写 | 必需 |
| C4 | `core/runtime.py:4335` `assemble` 的 inherit 分支 | 按 §6 的三级规则选源 | 必需 |
| C5 | `core/runtime.py:~4310` `assemble` 的 instantiate 调用 | `parent_agent_id=t.creator_agent_id` → `... or None` | 加固 |
| C7 | `lifecycle/agent_manager.py:378` 旁 | 新增 `assert_can_parent(agent_id)`：不存在 → `AgentNotFound`，`terminated` → `AgentTerminatedError`，其余（含 `running`）放行 | 必需 |

**C2 零迁移**：`deserialize_settings`（`task.py:73`）按 `__dataclass_fields__` 过滤，
存量事件缺这个键就落默认值。

**C3 的 keyword-only 不是洁癖**：第一个位置参数的类型从 `Task` 变成了 `str`，Python
不会为此报错——陈旧的位置调用会把一个 Task 对象静默绑到 `source_agent_id` 上，
`load_view` 拿它当 agent_id 查、查空，于是「继承了个寂寞」，只在下游断言处才隐约红。
实现时 `test_inherit_memory_snapshot.py` 的两处调用正是这样失败的。加一道 `*`
让这类调用在调用点就 `TypeError`。

**C5 是加固不是修复**：新入口预建 agent 后 `assemble` 走 `materialize`，够不到这条；
但 `instantiate` 的判据是 `if parent_agent_id is not None`——空串 `""` 不是 `None`，
会落进 `_register_fallback("")` 给一个不存在的 agent 建 record。今天 `delegate_task`
的 `creator_agent_id` 恒非空所以打不到，把它堵上是顺手的事。

**明确不做**：
- **不把 `send_message` 的新建分支收敛到 `dispatch_task`**（曾作为 C6 提出，已否决）。
  两者看着共用第 3/5/8/9/10/11 步，但那是形状相同、语义不同：`send_message` 是**对话**
  ——「对谁说话」由调用方给，「开新 task 还是并进现有 task」由 core 按 `current_task_id`
  路由；`dispatch_task` 是**派发**——开不开新 agent、挂谁、继承谁全由调用方声明，且
  **永不注入现有 task**。硬合的代价是把 `send_message` 的路由分支变成一个参数，让
  「决策归谁」这件事从签名里消失。两个入口各自直白，好过一个带模式开关的入口。
- 不暴露 `blocked_by`（`push_task` 支持依赖边，但「host 侧编排计划」是另一个题目，
  要做应当有自己的入口而不是塞进这个参数表）。
- 不动 `run_single_task`。它的处置（改成 `start_session` 薄包装 / 删除 + 25 处迁移）
  与本设计正交，单独决定。

---

## 9. 必须复刻的测试不变式

1. **新树建得出**：`agent_id=None, use_subagent=True` → 恰好一条新 `AgentInstantiated`、
   **零条 `AgentSpawned`**、新 agent `spawn_depth==0` 且 `parent_agent_id is None`、
   `Session.root_agent_id` 不变。
2. **派生建得对**：`agent_id=A, use_subagent=True` → `AgentSpawned{parent_agent_id: A}`
   先于 `AgentInstantiated`，新 agent `spawn_depth == A.spawn_depth + 1`。
3. **重放等价**（§3.1 的地基）：上面两例各自跑完后 `rebuild_view` 折出来的
   `AgentView.parent_agent_id` / `spawn_depth` 必须与内存里的 `_AgentRecord` 逐字段相等。
   两个方向都要断言，这是本设计最容易被后续改动悄悄打破的地方。
4. **森林不串味**：新树 agent 与 root 并发跑两条 task 时互不阻塞
   （`effective_agent_id` 给出不同串行键），且各自的 memory scope 不互相污染。
5. **跨树继承**：`agent_id=None` + `inherit_from_agent_id=<root>` → 新 agent 的
   起始记忆等于 root 当时的召回视图，且元数据带 `inherited_from_agent_id`。
6. **三级回落不回归**：不传 `inherit_from_agent_id` 时，`delegate_task` 派生的子任务
   仍从父任务的 agent 继承（现有 `test_inherit_mirror.py` 必须保持绿）。
7. **两道守卫分岔**（§7.2，本轮的核心决定，两个方向都要钉）：
   - `agent_id=A(running), use_subagent=False` → `AgentBusyError`，且**零副作用**
     （订阅总线断言事件数为 0、ALM 里没多出 agent、队列长度不变）
   - `agent_id=A(running), use_subagent=True` → **成功**，新子树建出来并开始跑，
     且 A 自己那条在跑的 task 不受影响（不被抢占、状态不变）
   - `agent_id=A(terminated)` → 两种 `use_subagent` 都抛 `AgentTerminatedError`
8. **派生不污染父对话**（§2.4）：`agent_id=A, use_subagent=True` 的 task 跑完后，
   A 的 agent scope 里**没有**派发框、没有 dispatch result；子树的 finish 对写在
   它自己的 scope 里。
9. **入口即拒**：携图内容 + `NullEventBlobStore` → `InvalidContentError`，且
   `AgentInstantiated` / `TaskCreated` 一条都没发（第 5 步先于第 7 步的判据）。
   V1–V4 四条纯参数校验各一例，同样断言零事件。
10. **不发 `SessionResumed` / 不受弃轮禁止**：会话里有在跑任务时派发成功，
    且事件流里没有 `SessionResumed`。
11. **句柄可寻址**：返回的 `handle.agent_id` 非空，且 `get_agent(handle.agent_id)` 查得到；
    task 跑完后对该 id `send_message` 能开出下一轮。

---

## 10. 与既有入口的分工

| 入口 | 语义 |
|---|---|
| `start_session` | 建一条**新会话**（或按 `resume=True` 开新一轮 root task，受弃轮禁止约束） |
| `send_message(agent_id, …)` | 对一个 agent **说话**：有活 task 就并进去，没有才开新 task。路由由 core 决定 |
| `dispatch_task(session_id, …)` | 在已有会话里**派发**一条顶层 task：开不开新 agent、挂谁、继承谁，全由调用方显式声明。**永不注入现有 task** |
| `delegate_task`（控制工具） | 同一件事的 LLM 侧对应物，但派的是**子任务**（有 `parent_task_id`，父任务挂起等它） |

一句话区分 `send_message` 与 `dispatch_task`：前者是**对话**，路由归 core；
后者是**派发**，决策归 host。

---

## 11. 容器会话：`create_session`

`dispatch_task` **不新建 session**（`session_id` 必填且须在事件日志里存在，冷复活
折不出 `SessionView` 就抛）。理由是它的整个语义建立在「root agent 不变」上，而一条
还不存在的会话里没有 root 可言：认第一条派发出去的 agent 当 root，这次调用就是
`start_session` 换了个名字；不认，就造出一条 `root_agent_id` 为空的会话，而
`resume_session` 明确会拒这种投影、恢复链上到处都假设它非空。

真正的缺口不在派发入口，在于**没有「建会话但先别跑」这个模式**——
`SessionRegistry.create_session` 恒推一条 root task。补上它：

```python
async def create_session(
    self, *,
    template_id: str,
    context_limit: int,
    session_id: str | None = None,
    tenant_id: str = "default",
    llm_account: str | None = None,
    llm_model: str | None = None,
    token_budget: int = 200_000,
    reserved_output_tokens: int = 8192,
) -> SessionHandle: ...
```

建会话、实例化 root agent、发 `SESSION_CREATED`，**不推 root task**。

**与 `start_session` 的分工只有一条：这一轮有没有「用户的话」。** `start_session` 收
`user_prompt` 并立刻拿它开一条对话式 root task；本方法没有 prompt 可收，建出来的会话
是空的、静止的、随时可以被派活。所以它**不收** `user_prompt` / `initial_task` /
`unattended`——那三个描述的都是 root task，而这里根本没有 root task。

**返回 `SessionHandle` 而不是 `TurnHandle`**：后者的四个身份字段恒非空、其中 `task_id`
是「这次交互落到的 task」，空会话里没有这样一条 task，硬造一个字段填不出来。
`SessionHandle` 是纯值对象（`session_id` / `root_agent_id` / `template_id`），刻意没有
`wait_for_finish()`——一件活都没有，没有可等的东西。要句柄就去 `dispatch_task` 拿。

**root agent 照常存在且 `idle`**，只是手上没有对话。「一个 session 恰有一个非空
`root_agent_id`」这条恢复链上的假设因此不破。之后两条路都成立：
`dispatch_task(..., agent_id=handle.root_agent_id)` 把活落在 root 头上，或者不传
`agent_id` 另起一棵树。

### 11.1 两条容易踩空的接线

**① TM 必须连 runner 一起接好**（走与 `start_session` 同一个 `_register_and_drain`）。
不接的话 `dispatch_task` 末尾那次 `drain()` 会撞
`RuntimeError("No task runner registered")`——而它是 `asyncio.create_task` fire-and-forget
出去的，撞了只在日志里，派进来的 task 就此静默搁浅在 `PENDING`。§9 的容器会话用例
用变异检验钉过这一条（摘掉 `set_runner` 后必红）。

**② 空队列不得自己宣告会话结束。** 这一点是既有实现白送的，但必须钉住：`drain()` 在
空队列上只是 `break` 出循环，会话终结信号只从 `on_task_finished`（且 `is_done()`）/
`finalize_idle_session` / `cancel_all` 发出。容器会话建出来后不发
`SessionFinished`，TM 保持 `is_alive()`。

### 11.2 冷路径

容器会话跨进程重启后被 `dispatch_task` 派活，走的是第 3 步的冷复活
（`_recover_session_locked(keep_alive=True)`）。零 task 的会话正是 `keep_alive` 的两处
短路已经预留的情形——「空历史在这条调用路径上是合法起点，不是损坏投影」
（`runtime.py` 该处注释），不必新增分支。

### 11.3 实现

| # | 位置 | 改动 |
|---|---|---|
| C8 | `lifecycle/session_registry.py` `create_session` | 加 `with_root_task: bool = True`；为假时返回 `(session, None, tm)`，跳过 `_make_root_task_manager` |
| C9 | 同上，新增 `_new_task_manager(session)` | TaskManager 的**唯一构造点**（容器分支与 `_make_root_task_manager` 共用），构造时一并 `set_session`——晚注入会让第一条 `TaskCreated` 落到 default 租户（总账 A5） |
| C10 | `core/runtime.py` 新增 `create_session` + `SessionHandle` | 见上；`SessionHandle` 从 `ctx_weft` 导出 |
