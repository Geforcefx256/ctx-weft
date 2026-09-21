# 宿主迁移清单：切到 ctx-weft 自带的 `SqlMemoryProvider`

日期：2026-08-29
适用版本：ctx-weft `feat/multimodal`（`providers/` 目录重组 + blob 归属 + 事件持久化落地之后）
相关设计：
[多模态整体设计 §14](superpowers/specs/2026-08-20-multimodal-design.md)、
[providers 目录 / SQL 事件存储设计](superpowers/specs/2026-08-29-providers-layout-and-sql-event-store-design.md)

> **这份文档是给人执行的**，不是给 agent 读的。按顺序做，每一步都有验收判据。
> 用户裁定 D4 是「宿主后续用这个 memory，平滑切换」——**平滑指存量行零迁移**，
> 但**不指零 DDL**：第 2 步里有一条**破坏性 DDL 必须先做**，只加列绕不开。
>
> 本次重写相对旧版最大的变化：**blob 字节已经离开 SQL memory provider**。
> 旧版教你「注册 `SqlMemoryProvider` 即自动获得 blob 能力」，那条自动解析路径
> 已经不存在——`SqlMemoryProvider` 现在只维护引用边，字节必须单独接一个
> `BlobStore` 实现（本仓自带 `FsBlobStore` 可直接用）。第 4、5 步已按新接线整体
> 重写；其余步骤与旧版一致。

---

## ⚠️ 实施回执（2026-09-21，NetliveCoworkPy 实际切完之后补记）

三处「照手册做会踩到」的，按严重程度排：

1. **「存量行零迁移」只在宿主的租户全是 `default` 时成立。** §2.2 那句话是本文最危险的
   一句：真有租户的宿主（NetliveCowork 的 `session.tenant_id` 是
   `"{surrogate_id}:{cowork_id}"`，且 tenant 随请求进 `ProviderContext`）切过去之后，
   老会话的每个读都是 `COALESCE(tenant,'default') = '<真租户>'`，而存量行归一成
   `'default'` —— **`load_view` 返回空列表，不报错**，表现是历史对话没了。
   **必须**按宿主自己的会话表回填一次 `memory_events.tenant` / `memory_subscriptions.tenant`
   （参考实现：该仓的 `m023_memory_tenant_and_content_format`）。
   零迁移的只有「租户本来就是 default」那部分。
2. **`tenant` 列别照抄 `String(64)`。** 本仓模型声明的是 64，而形如
   `"{surrogate_id}:{cowork_id}"` 的租户 id 轻易超过它：Postgres 直接
   `value too long for character varying(64)`，SQLite 不认长度、静默存进去——于是本地
   测试全绿、上生产必炸。建表 DDL 归宿主，按自己的租户 id 长度放宽即可。
3. **回填 `content_format` 时，判据要比 §2.4 给的那条严一档。** §2.4 写的是照搬读侧
   启发式（`json.loads` 出 list 即 `'parts'`），但读侧解 parts 走
   `content_from_jsonable`，它对每个元素调 `.get("type")`——正文恰为 `"[1, 2]"` 的行判成
   `'parts'` 就是给它安排一次 `AttributeError`，炸在装配路径上。判据改成
   **「list 且每个元素都是 dict」**：真正的多模态存量行照旧判 `'parts'`，`[1, 2]` 落
   `'text'`（原样以 JSON 文本呈现）。「读得到、是文本」优于「一读就炸」。

另有两条不是坑：

- 第 4、5 步与第 7 步的坑，对**已经接完 blob** 的宿主是空操作——先核对再动手，别重复实现。
- 第 4 步「强烈建议分开部署两个实例」那句，是给「不愿维护事件侧活引用枚举」的宿主的逃生
  门。**有第三处引用边的宿主根本走不进这扇门**，该反过来选共用一个实例：NetliveCowork 的
  帧日志（携图消息在 SSE 帧里也只留 ref）是它自己造的引用边，core 不知情，所以它的事件侧
  回收一直喂的就是「事件 payload ∪ 帧日志」——并集的维护成本早就在付了，分开只是让它在
  付着同样成本的同时，把同一张图在两个 root 各存一份（core 外部化进 memory 侧、宿主写帧
  日志时又 put 进 event 侧），读路径还得两侧都试。它已按这个方向合并成一个实例、一个目录，
  回收改喂一个 `all_live_blob_refs()`（三处并集，单一维护点）。

---

## 0. 这次切换会得到什么 / 失去什么

**得到**：

- **多租户隔离**。宿主现有的 `postgres.py` 参考实现**完全没有 tenant**：
  `load_view` / `recall_topic` / `list_subscriptions` 都不看 `ctx.tenant_id`，
  PUBLICATION 覆盖按 topic 扫全表。这四个洞在 ctx-weft 侧已实证并修复
  （见设计 §14.2），其中一个是**写侧**的——「B 一发布就把 A 的黑板行标成
  superseded，而 B 自己根本读不到那行」，即**能销毁自己读不到的数据**。
- **多模态能真的存住**。参考实现的 `_ingest_in_tx` 里是 `json.dumps(event.content)`，
  而 `ContentPart` 是普通 dataclass —— **一旦 content 带图就 `TypeError`**（今日不炸
  只因 content 全是 `str`）。新 provider 有判别列 + jsonable 往返。
- **blob 引用边与延迟回收**——字节本身由宿主单独接的 `BlobStore` 实现持有
  （见第 4、5 步），`SqlMemoryProvider` 只管 `memory_blob_refs` 那张引用边表。

**失去 / 需要留意**：

- `FilesystemBlobStore` 已被移除（裁定 D5）。若宿主此前显式
  `register_blob_store(FilesystemToolsProvider(...))`，那行代码会失效。
- **`SqlMemoryProvider` 不再实现 `MemoryBlobStore` / `EventBlobStore`。** 旧版文档
  说的「`ProviderRegistry.get_blob_store()` 会自动解析到 memory provider」这条路径
  已经删除——注册 `SqlMemoryProvider` **不会**顺带打开图片外部化，必须显式注册
  blob store（见第 4 步）。没注册就是「不接 blob」：携图会话会在入口被
  `BlobStoreRequiredError` 拒绝。
- 文末 legacy 方法（`recall_recent` / `recall_recent_by_agent` / `count_recent` /
  `supersede`）**新 provider 不实现**——它们已被 P4b-2 移出协议。宿主若有直调点，
  必须先改成协议面的 `load_view` / `fold`（见第 7 步）。

---

## 1. 依赖

```bash
pip install "ctx-weft[sql]"        # sqlalchemy>=2.0 + aiosqlite>=0.19
```

`aiosqlite` 只有 SQLite 后端需要；接 postgres 的宿主自带 `asyncpg`/`psycopg` 即可。

**这是可选依赖，且没有任何上层包 eager import 它**——缺 sqlalchemy 时
`import ctx_weft` / `import ctx_weft.providers` 照常工作，只有显式
`import ctx_weft.providers.memory.sql` 才会如实报 `ImportError`。

---

## 2. DDL（**先做这一步，且第 2.1 条必须最先做**）

### 2.1 ⚠️ 必须先 DROP 旧的三列唯一索引

宿主现有：

```sql
-- 旧：3 列，不含 tenant
ix_subscriptions_session_task_topic  ON memory_subscriptions (session_id, task_id, topic)
```

新 provider 的订阅幂等键**必须含 tenant**（隔离契约第 4 条：否则同 `session_id` 的
另一个租户会撞进幂等分支、拿到别人的订阅对象与游标）。若不 DROP 旧索引，
**两个租户的同 `(session, task, topic)` 订阅会撞 `IntegrityError`**——
这不是「只加 nullable 列」能覆盖的，是本次迁移唯一一条破坏性 DDL。

```sql
-- postgres：先确认它是索引还是约束
--   \d memory_subscriptions      （psql）
-- 若是 UNIQUE INDEX：
DROP INDEX IF EXISTS ix_subscriptions_session_task_topic;
-- 若是 UNIQUE CONSTRAINT（名字相同也可能是约束）：
ALTER TABLE memory_subscriptions DROP CONSTRAINT IF EXISTS ix_subscriptions_session_task_topic;
```

**验收**：`ix_subscriptions_session_task_topic` 在系统表里查不到。

### 2.2 加列（全部 nullable，存量行零迁移）

```sql
ALTER TABLE memory_events        ADD COLUMN tenant         VARCHAR(64) NULL;
ALTER TABLE memory_events        ADD COLUMN content_format VARCHAR(16) NULL;
ALTER TABLE memory_subscriptions ADD COLUMN tenant         VARCHAR(64) NULL;
```

**不要给它们加 NOT NULL 或 DEFAULT**。`NULL` 在两个列上各有确定语义：

- `tenant IS NULL` ≡ `'default'` 租户。读侧一律 `COALESCE(tenant,'default')`，
  与 core 侧的 `normalize_tenant(t) = t or "default"` 是同一条规则的 SQL 写法，
  **存量行不迁移即落默认分区**——⚠️ 这句只在你的租户全是 `default` 时成立，
  否则必须回填，见文首「实施回执」第 1 条（不回填 = 历史记忆静默消失）。
  列长度也别照抄 `String(64)`，见回执第 2 条。
- `content_format IS NULL` **唯一地表示「存量行」**（见 2.4）。

### 2.3 建索引与两张新表

```sql
CREATE UNIQUE INDEX ix_subscriptions_tenant_session_task_topic
    ON memory_subscriptions (tenant, session_id, task_id, topic);

CREATE INDEX ix_memory_tenant_task  ON memory_events (tenant, session_id, layer, task_id);
CREATE INDEX ix_memory_tenant_agent ON memory_events (tenant, session_id, layer, agent_id);
CREATE INDEX ix_memory_tenant_topic ON memory_events (tenant, topic, topic_seq_no);

-- 事件 → blob 的引用边。只存 sha 引用，不存字节；刻意无外键（见设计 §14.4）。
CREATE TABLE memory_blob_refs (
    event_id VARCHAR(64) NOT NULL,
    sha      VARCHAR(64) NOT NULL,
    PRIMARY KEY (event_id, sha)
);
CREATE INDEX ix_memory_blob_refs_sha ON memory_blob_refs (sha);
```

**没有 `memory_blobs` 表了。** 旧版本这里还建了一张持有字节的 `memory_blobs`
（`sha` / `media_type` / `data BYTEA`），那是 blob 字节还留在 RDBMS 时代的产物。
现在 `SqlMemoryProvider` 只维护上面这张纯引用边表，字节由第 4 步单独接的
`BlobStore` 实现（如 `FsBlobStore`）持有，不进这个库。

`ix_memory_task` / `ix_memory_agent` 两个既有索引**保持原样，名字与列都不要动**——
新 provider 的模型定义与宿主逐字一致。

**验收**：`SqlMemoryProvider` 的 `models.py` 与库里的实际 schema 对得上。
SQLite 单机部署可以直接用 `open_sqlite_memory(path)`，它会 `create_all`。

### 2.4 建议：给存量行回填 `content_format`

**不回填也能跑**，但存量行会继续走一条**启发式**：读侧对 `content_format IS NULL`
的行「试 `json.loads`，解出 list 就当结构化内容」。后果：**一条正文恰好是
`"[1, 2]"` 的存量纯文本消息会被误读成 parts**。这是继承自宿主参考实现的既有歧义
（新行已无此问题——新行一律写非空判别列，`"text"` 或 `"parts"`，三态而非两态）。

彻底治法是迁移时显式回填一次（⚠️ 判据要比下面这条严一档，见文首「实施回执」第 3 条）：

```sql
-- 只有当你确信存量 content 全是纯文本时（Phase 3c 之前的宿主正是如此）：
UPDATE memory_events SET content_format = 'text' WHERE content_format IS NULL;
```

若宿主存量里**确实**有 pre-E2 的 `json.dumps(list-of-dict)` 行，先用
`WHERE content LIKE '[%'` 挑出来人工核对，再分别回填 `'parts'` / `'text'`。

**验收**：`SELECT count(*) FROM memory_events WHERE content_format IS NULL` 为 0
（或为「你确认要留给启发式的那部分」）。

---

## 3. 接线：换 memory provider

```python
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from ctx_weft.providers.memory.sql import SqlMemoryProvider

engine  = create_async_engine(dsn)                      # 宿主自己的 engine
factory = async_sessionmaker(engine, expire_on_commit=False)
memory  = SqlMemoryProvider(factory)

runtime.providers.register_memory(memory)
```

注意包路径是 `ctx_weft.providers.memory.sql`（不是旧版的 `providers.memory_sql`——
`providers/` 已按「领域 → 协议 → 变体」重组，全部 provider 的 import 路径都变了，
不留兼容 shim）。

宿主接 postgres 时**自带 engine 与 migration**，直接构造 `SqlMemoryProvider(factory)`
即可，不必走 `open_sqlite_memory`（那是测试与单机部署用的便利函数，会 `create_all`）。

**验收**：跑一遍宿主自己的 memory 冒烟用例；ctx-weft 侧的协议一致性套
`tests/unit/test_memory_conformance.py` 是同一批断言（80 条 × provider）。

---

## 4. ⚠️ 接线：blob 字节必须单独注册

**`SqlMemoryProvider` 不实现 `MemoryBlobStore` / `EventBlobStore`。** 它只维护
`memory_blob_refs` 这张引用边表（这部分必须与 ingest 同事务，所以留在 SQL 侧），
字节的存取归 blob store 单独负责。`ProviderRegistry` 上 blob 相关的两个 getter
**都不会**自动解析到 memory provider——`get_memory_blob_store()` 与
`get_event_blob_store()` 完全对称，各自只有两级：显式注册 > `Null*BlobStore`。
没注册就是「不接 blob」，携图会话会在入口被 `BlobStoreRequiredError` 拒绝。

接线用本仓自带的 `FsBlobStore`（文件系统内容寻址实现，可直接用于生产，也可以
换成宿主自己的对象存储实现）：

```python
from pathlib import Path
from ctx_weft.providers.blob.fs import FsBlobStore

blobs = FsBlobStore(Path("/var/lib/app/blobs"))
runtime.providers.register_memory_blob_store(blobs)
runtime.providers.register_event_blob_store(blobs)   # 共用一个实例是允许的
```

`FsBlobStore` 同时实现 `MemoryBlobStore` 与 `EventBlobStore` 两个协议，是「一个类
满足两个契约」的示例——但**两个协议各自独立定义、语义会各自演进**，共用一个实例
只是这份实现恰好两边都能用，不代表两个协议本身合并了。

⚠️ **两个协议的 ref 命名空间彼此独立**，core 从不比较、也从不拿一侧的 ref 去
另一侧解析。宿主也可以分开部署两个 `FsBlobStore` 实例（各指向不同目录）分别注册
为 memory 侧与 event 侧——这样第 5 步的回收陷阱自动不存在。

**验收**：接线后 `runtime.providers.get_memory_blob_store()` /
`get_event_blob_store()` 返回的是你注册的 `blobs` 实例（不是 `NullMemoryBlobStore`
/ `NullEventBlobStore`）；携图会话不再在入口报 `BlobStoreRequiredError`。

---

## 5. ⚠️ 回收变成标准 mark-sweep 两步，必须由宿主自己定时调

旧版本这里是一步 `await memory.collect_blobs()`——那是 blob 字节还存在
`SqlMemoryProvider` 自己那张 `memory_blobs` 表里的时代。现在字节已经搬到独立的
blob store，回收自然拆成标准的 mark-sweep 两步：**mark 在 memory**（它持有引用
边，知道哪些 sha 还被活记录引用），**sweep 在 blob store**（它持有字节，知道怎么
删）。

```python
# 例：每小时一次的后台任务
live = await memory.live_blob_refs()
deleted = await blobs.collect(live)
```

- `live_blob_refs() -> set[str]`（`SqlMemoryProvider` 的自有方法，不在
  `MemoryProvider` 协议里）返回当前仍被 `is_superseded = 0` 的记录引用的全部
  `blob:<sha>`。
- `collect(live_refs, *, now=None) -> int`（`BlobStore` 双协议都声明）把不在
  `live_refs` 里、且已过宽限期（`FsBlobStore` 默认 **24 小时**，构造时
  `FsBlobStore(root, grace_period=timedelta(hours=N))` 可调）的字节删掉，返回删除数。
- **两条都是幂等的**，可随时重跑，可并发重入（多删一次也只是删不到）。
- **宽限期是正确性要求，不是优化**：`put` 与 `ingest`/事件落库之间存在时序窗口
  （进程内是毫秒级，但中间可能隔着 HITL park——那能等人数小时）。窗口里的 blob
  在 `live_blob_refs()` 里查不到（还没有引用边），没有宽限期就会被清扫误删，
  图**永久丢失**。
- **不调用回收的后果是「blob 只涨不删」**（泄漏磁盘），**不会**产生悬空 ref。
  即：忘了配定时任务不会坏数据，只会费磁盘。

### ⚠️ 共用一个 blob store 实例时的陷阱

如果第 4 步里 memory 侧与 event 侧注册的是**同一个** `blobs` 实例（如上面的
写法），回收时喂给 `collect()` 的 `live_refs` **必须同时包含两侧的活引用**——
只喂 memory 侧会把事件流仍需要的字节当孤儿删掉：

```python
# 错——只喂了 memory 侧，会把事件流仍需要的字节当孤儿删掉
deleted = await blobs.collect(await memory.live_blob_refs())
```

**`memory.live_blob_refs()` 是 `SqlMemoryProvider` 的自有方法，`EventStore`
协议没有对应物**——ctx-weft **不提供**「枚举一个 event store 当前活引用」的
现成接口（事件本就不像 memory 记录那样有 `is_superseded` 语义，「哪些事件仍然
有效」是宿主的保留策略，不是 core 能替你判断的）。

真要自己算，ref 藏在**事件的 `payload`** 里——由 `content_to_event_jsonable`
（`core/content.py`）在五个参与状态重建的事件发射点（`SESSION_CREATED` /
`SESSION_RESUMED` / `TASK_CREATED` / `TASK_REQUEUED` / `HITL_*`）写入，**键名随
事件类型而异**（例如 `SESSION_CREATED`/`SESSION_RESUMED` 是 `payload["user_prompt"]`，
`HITL_ANSWERED` 等 HITL 事件是 `payload["message"]`）。要从中提取 `blob:<sha>`，
得先用 `core.content.content_from_jsonable` 把该键的值还原成 `list[ContentPart]`，
再喂给 `core.content.extract_blob_refs`——但**该取哪个键，得宿主自己按事件类型
判断**，ctx-weft 没有替你把这层封装成一个函数。也就是说，**共用一个实例时，
正确的并集要靠宿主自己维护这套「按事件类型取键 → 还原 → 提取 ref」的逻辑**，
没有便利函数可用。

正因为「共用一个实例」时回收的正确性要靠宿主自己维护上面这套事件侧活引用枚举
逻辑，**除非你愿意维护它，否则强烈建议改为分开部署两个独立的 blob store 实例**
（各指向不同目录/桶）：memory 侧回收只看 `memory.live_blob_refs()`，event 侧按
宿主自己的事件保留策略（例如「保留最近 N 天」，过期即整体归档/删除，根本不需要
按 sha 级别回收）单独处理，两侧从此互不干扰，也不需要写那套枚举逻辑。

**验收**：定时任务上线后观察 blob store 的存储用量不再单调上涨；若共用一个实例，
额外确认回收前后携图会话仍能在崩溃恢复后正确显示图片。

---

## 6. 事件持久化也可以换成 `SqlEventStore`

memory 换 SQL 之后，事件流也可以从默认的 `InMemoryEventStore` 换成
`SqlEventStore`（需同一个 `ctx-weft[sql]` 依赖），得到跨进程重启不丢事件、
支持崩溃恢复快照的持久化。

单文件 SQLite 部署可以与 memory 共享同一个 `engine`/`factory`，各自 `create_all`
一次（两个包的 `Base` 是独立的——`memory.sql` 一个、`events.store.sql` 一个，
共用会强迫只想建 events 表的宿主连 memory 表一起建）：

```python
from ctx_weft.providers.events import attach_persistence
from ctx_weft.providers.events.store import sql as events_sql
from ctx_weft.providers.events.store.sql import SqlEventStore
from ctx_weft.providers.memory import sql as memory_sql
from ctx_weft.providers.memory.sql import SqlMemoryProvider, make_session_factory

engine, factory = make_session_factory("sqlite+aiosqlite:///app.db")
async with engine.begin() as conn:
    await conn.run_sync(memory_sql.Base.metadata.create_all)
    await conn.run_sync(events_sql.Base.metadata.create_all)

memory = SqlMemoryProvider(factory)
store = SqlEventStore(factory)

# ⚠️ store 必须在 CtxWeftRuntime 构造之前建好，并作为 event_store= 传入——
# 不是建完 runtime 之后再对 runtime.event_bus 调 attach_persistence()。
# CtxWeftRuntime.__init__ 内部自己调用 attach_persistence(bus, event_store,
# snapshot_every_n=...)；recover() / rebuild_view / read_session_events_of_types
# 等**所有读路径**读的都是 self.event_store。如果晚建 runtime、早建 store 再补接，
# runtime 内部默认的 InMemoryEventStore 早已经是被 attach 好的那份，你自己接的
# store 只会被写、从不会被读——重启后 recover() 从空的内存 store 恢复，"崩溃后
# 存活"这个工作流的目的根本不工作，还会让每条事件存两遍。
runtime = CtxWeftRuntime(
    providers=registry, event_store=store, snapshot_every_n=50)
runtime.providers.register_memory(memory)
```

宿主接 postgres、自带 engine/migration 时同理：直接 `SqlEventStore(factory)`，
不必走 `open_sqlite_event_store`（那是单机部署用的便利函数）。

要点：

- **`CtxWeftRuntime(event_store=..., snapshot_every_n=...)` 是唯一推荐的接线方式**。
  `attach_persistence` 在 `__init__` 内部被调用，它保证 `EventPersister` 先于
  `SnapshotWriter` 订阅（顺序反了会让快照的 `last_event_id` 与它实际看到的
  view 对不上）。不要在 runtime 构造之后再自己额外调一次 `attach_persistence`
  ——那样等于对同一条 bus 挂了两个 persister，`SqlEventStore` 那份会因为
  `events.id` 主键冲突每条事件抛一次 `IntegrityError`（被吞成
  `logger.exception`，日志刷屏但不掀 loop）。只有在完全不经过 `CtxWeftRuntime`
  独立使用事件总线时（例如单测 store 往返），才直接调 `attach_persistence`。
- `runtime.persistence` 是 `attach_persistence` 返回的 `PersistenceHandle`（公开
  属性），需要换 store 或停止持久化时调 `await runtime.persistence.detach()`。
- `snapshot_every_n=0`（默认）等价于「不接快照」，现有行为零变化；传正数
  才会额外接上 `SnapshotWriter`，把崩溃恢复从 O(全部事件) 全量回放降级成
  「最新快照 + 增量」。
- `SqlEventStore` 与 `InMemoryEventStore` 在 `append()` 上同口径：**都不过滤瞬态
  事件**，过滤是 `EventPersister` 的订阅策略，不是存储策略。若宿主此前直接调
  `event_store.append()`，注意每 token 一个的流式 delta 现在会真的落库。

**验收**：重启进程（或在测试里模拟"新建一个 `CtxWeftRuntime` 但传入同一个
`factory` 建出的 `SqlEventStore`"）后，`await runtime.recover()` 能从这个
`SqlEventStore` 里读回此前落盘的事件、恢复出正确的 session 状态——而不是仅仅
观察到事件被写了进去。只验"写"不验"重启后能读回"查不出接反的问题（读永远
走的是 runtime 内部默认的 `InMemoryEventStore`，重启后就是空的，但『冒烟结果
与 InMemoryEventStore 一致』这个说法在这种接反状态下照样为真）。ctx-weft 侧的
一致性套 `tests/unit/test_event_store_conformance.py` 用同一套用例跑
`InMemoryEventStore` 与 `SqlEventStore` 的 store 级往返，可作为参考基线，但
它不覆盖"接入 `CtxWeftRuntime` 后重启恢复"这条——那条需要宿主自己验。

---

## 7. 宿主自己的 event model 做多模态改造时的五个坑

宿主的参考实现 `providers/memory/postgres.py`（462 行）落后于当前协议七处，
下面五条是真会咬人的（另两处是 legacy 方法与 `recall_topic` 语义，见第 8 步）：

1. **`json.dumps(event.content)` 对 dataclass 直接 `TypeError`。** 必须走
   `ctx_weft.core.content.content_to_jsonable` / `content_from_jsonable`。
2. **读侧不要靠猜。** 参考实现「试 `json.loads`，是 list 就当结构化内容」会把正文
   恰为 JSON 数组的用户消息误判。新行**一律写非空判别列**（`"text"` / `"parts"`），
   `NULL` 因此唯一表示存量行——**两态改三态**是这次的关键设计。
3. **SQLite 会静默丢 tzinfo。** 裸 `DateTime(timezone=True)` 在 SQLite 上回读得到
   naive datetime，排序照常工作，但**等值比较恒 False**。新 provider 包了一层
   `UtcDateTime` TypeDecorator（在 postgres 上是恒等变换，`memory.sql` 与
   `events.store.sql` 各自从共享的 `providers/_sqlalchemy.py` 引用同一实现）。
   宿主若共用模型文件，直接用它。
4. **tenant 必须进列 + 索引/唯一键**，不能只在应用层过滤。写读两侧用**同一条**
   归一规则（`COALESCE(tenant,'default')`）。
5. **`MemoryEventModel` 与宿主的 `SessionModel` / `TaskModel` / `EventModel`
   共用同一个 `Base`。** 拆开意味着两个注册表、`create_all` 各建各的；宿主的
   `observability/session_export.py` 与 `session_import.py` 都直接 import 了
   `MemoryEventModel`，会被波及。**这是本次迁移最大的一块工程摩擦，且不在
   ctx-weft 仓内**——建议单列一个任务，并先做只读核对（列清楚哪些文件会被波及）
   再动手。

---

## 8. 迁移前必须先改掉的宿主调用点

- **legacy 方法**：`recall_recent` / `recall_recent_by_agent` / `count_recent` /
  `supersede` **新 provider 不实现**（P4b-2 已移出协议）。宿主有直调点的话，
  改成 `load_view(address, scope, ctx, kinds=...)` 与
  `fold(supersede_ids, replacements, ctx)`。
  顺带一提：这些方法里的 tenant 比较**本来就是空转**（等式两边都用读侧 ctx 的
  tenant，约掉了），所以它们**没有**任何隔离——不改而是继续用，等于把已修好的洞
  留一扇后门。
- **`recall_topic` 不比 `session_id`**（新旧两个实现一致、协议未定）。这意味着
  **同租户内跨 session 的同名 topic 是共享的**。宿主若自定过 `long_term_*` 之类的
  固定 topic 名，切换前请确认这正是你要的语义（详见设计 §14.7 的 L25）。

---

## 9. 验收清单（照着勾）

- [ ] `pip install "ctx-weft[sql]"` 完成
- [ ] `ix_subscriptions_session_task_topic`（3 列）**已 DROP**
- [ ] `memory_events.tenant` / `memory_events.content_format` /
      `memory_subscriptions.tenant` 三列已加，且**均为 nullable、无 DEFAULT**
- [ ] `ix_subscriptions_tenant_session_task_topic`（4 列 unique）已建
- [ ] `ix_memory_tenant_task` / `_agent` / `_topic` 已建
- [ ] `memory_blob_refs` 表已建（含 `ix_memory_blob_refs_sha`）——**没有**
      `memory_blobs` 表，字节不进这个库
- [ ] 存量行 `content_format` 已回填（或已确认不回填的风险）
- [ ] `register_memory(SqlMemoryProvider(factory))` 已接
- [ ] `register_memory_blob_store(...)` / `register_event_blob_store(...)`
      **均已显式注册**（不注册就是没有 blob 能力，不是自动获得）
- [ ] 若两侧共用同一个 blob store 实例，回收脚本的 `live_refs` 已确认取的是
      两侧并集；若分开部署两个实例，两个回收任务已分别配置
- [ ] 回收（mark-sweep 两步）的定时任务已配
- [ ] 宿主对 legacy 四方法的调用点已清零
- [ ] 两个租户用**同一个 `session_id`** 跑一遍冒烟：A 读不到 B 的行，
      B 发布不会把 A 的行标 superseded，两边订阅各自独立
- [ ] （若同时切换事件持久化）`store = SqlEventStore(factory)` 已在
      `CtxWeftRuntime(event_store=store, ...)` 构造**之前**建好并传入——**不是**
      建完 runtime 后再对 `runtime.event_bus` 补调一次 `attach_persistence`；
      重启进程（或用同一个 `factory` 重新构造一个 `CtxWeftRuntime`）后
      `await runtime.recover()` 确认能从这个 `SqlEventStore` 里读回事件、恢复出
      正确的 session 状态（只验证事件被写入不够——接反时写入也会成功，只是
      读路径读的是另一个从未被读过的 `InMemoryEventStore`，重启后就是空的）

---

## 10. 回滚

DDL 全是加列/加索引/加表（除 2.1 那条 DROP），回滚方向：

1. 代码切回旧 provider，并把 `register_memory_blob_store` / 
   `register_event_blob_store` 的接线也一并撤掉（或换回旧的
   `register_blob_store(FilesystemToolsProvider(...))`，如果宿主当时是这样接的）；
2. 重建旧的 3 列唯一索引——**但先确认此时库里没有「同 (session,task,topic)
   不同 tenant」的订阅行**，有的话重建会失败，需要先决定保留哪一条；
3. 新增的三列、`memory_blob_refs` 表可以留着不管（旧 provider 不读它们），
   也可以 DROP。

**不可回滚的一件事**：切换期间新写入的图片，字节只在 blob store（如
`FsBlobStore` 指向的文件系统目录）里，事件行/memory 行里只有 ref。回滚到旧
provider 后那些图取不回来（旧 provider 不认识 ref，也不知道去哪个目录找字节）。
