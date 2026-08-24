# Architecture Diagram

The system at a glance. Each diagram below answers one of the questions the brief asks:
**what services, what talks to what, what compute runs where, and where the data lives.**

The reasoning behind every choice — why LangGraph, why Gemini tiered, why BigQuery
`VECTOR_SEARCH` before Vertex AI Vector Search — is in
**[the design document](HLD.md)**, which also carries six further diagrams covering
retrieval, the learning loops, failover and the evaluation pyramid.

| # | Diagram | Question it answers |
|---|---|---|
| 1 | [System architecture](#1-system-architecture) | What services, and how do they communicate? |
| 2 | [Compute and request path](#2-compute-and-request-path) | What compute runs where, and how does it scale? |
| 3 | [Where the data lives](#3-where-the-data-lives) | How and where is data stored and handled? |
| 4 | [The agent graph](#4-the-agent-graph) | What is the control flow inside a turn? |
| 5 | [Anatomy of a turn](#5-anatomy-of-a-turn) | What is the end-to-end data flow? |
| 6 | [Safety enforcement points](#6-safety-enforcement-points) | Where is policy actually enforced? |

---

## 1. System architecture

Building blocks, services and the communication between them. Red is the safety kernel;
blue is persistent storage.

```mermaid
flowchart TB
    subgraph clients["Client layer"]
        CLI["CLI chat<br/><i>the prototype</i>"]
        SLACK["Slack bot"]
        WEB["Web UI"]
    end

    subgraph edge["Edge"]
        LB["Cloud Load Balancing<br/>+ Cloud Armor (WAF, rate limits)"]
        IAP["Identity-Aware Proxy<br/><i>SSO, group -> role</i>"]
    end

    subgraph runtime["Agent runtime — Cloud Run"]
        API["FastAPI gateway<br/><i>SSE streaming, auth, quotas</i>"]
        ORCH["LangGraph orchestrator<br/><i>the agent graph, §3</i>"]
        SAFE["Safety kernel<br/><i>guardrail · SQL validator · PII masker</i>"]
        TOOLS["Tool registry<br/><i>warehouse · reports · charts · email · web</i>"]
    end

    subgraph models["Model layer — Vertex AI"]
        FLASH["Gemini 2.5 Flash<br/><i>guardrail · plan · SQL · repair</i>"]
        PRO["Gemini 2.5 Pro<br/><i>analysis · reports · judge</i>"]
        EMB["gemini-embedding-001"]
        FB["Fallback chain<br/><i>AI Studio -> OpenRouter -> self-hosted</i>"]
    end

    subgraph data["Data layer"]
        BQ[("BigQuery<br/><i>thelook_ecommerce</i><br/>read-only SA + authorized views")]
        GCS[("GCS<br/><i>golden trios · personas<br/>· report exports</i>")]
        VEC[("Vector index<br/><i>BQ VECTOR_SEARCH -><br/>Vertex Vector Search at scale</i>")]
        FS[("Firestore<br/><i>sessions · checkpoints · user profiles<br/>· saved reports · audit log</i>")]
        REDIS[("Memorystore<br/><i>semantic + schema cache</i>")]
        SM[("Secret Manager")]
    end

    subgraph async["Asynchronous"]
        PS["Pub/Sub"]
        CT["Cloud Tasks<br/><i>long reports, scheduled digests</i>"]
        CF["Cloud Functions<br/><i>trio indexing · config validation</i>"]
    end

    subgraph obs["Observability"]
        OTEL["OpenTelemetry SDK"]
        TRACE["Cloud Trace"]
        LOGS["Cloud Logging -> BigQuery sink"]
        MON["Cloud Monitoring<br/><i>SLIs, alerts -> PagerDuty</i>"]
    end

    CLI & SLACK & WEB --> LB --> IAP --> API
    API --> ORCH
    ORCH <--> SAFE
    ORCH --> TOOLS
    ORCH --> FLASH & PRO
    FLASH & PRO -.on failure.-> FB
    ORCH --> EMB --> VEC
    TOOLS --> BQ
    TOOLS --> FS
    ORCH <--> FS
    ORCH <--> REDIS
    GCS --> CF --> VEC
    GCS -.personas, hot reload.-> ORCH
    ORCH --> PS --> CF
    API --> CT --> ORCH
    SM -.credentials.-> ORCH
    ORCH --> OTEL --> TRACE & LOGS & MON

    classDef safety fill:#7f1d1d,stroke:#fca5a5,color:#fff
    classDef store fill:#1e3a8a,stroke:#93c5fd,color:#fff
    class SAFE safety
    class BQ,GCS,VEC,FS,REDIS,SM store
```

Detail: [HLD §2](HLD.md#2-system-architecture) · [service choices](HLD.md#5-technology-choices-and-why)

---

## 2. Compute and request path

What actually runs, on what, and what happens when it scales. Everything on the synchronous
path is one Cloud Run container; the write-behind work is deliberately off it.

```mermaid
flowchart LR
    subgraph sync["Synchronous path — Cloud Run, autoscaled 1..N"]
        direction TB
        REQ["HTTPS + SSE<br/><i>token streaming</i>"] --> FA["FastAPI gateway<br/><i>auth · quotas · rate limit</i>"]
        FA --> LG["LangGraph orchestrator<br/><i>in-process, ~200 MB</i>"]
        LG --> SK["Safety kernel<br/><i>pure CPU, no I/O</i>"]
        LG --> IO["I/O fan-out<br/><i>models · warehouse · state</i>"]
    end

    subgraph async["Write-behind — never blocks the answer"]
        direction TB
        PS["Pub/Sub"] --> CF["Cloud Functions<br/><i>trio indexing · config validation</i>"]
        CT["Cloud Tasks"] --> LR["Long reports<br/><i>4+ queries, Pro model</i>"]
        SCH["Cloud Scheduler"] --> NB["Nightly evals<br/>+ quality decay"]
    end

    IO -.-> PS
    FA --> CT

    subgraph scale["Scaling characteristics"]
        direction TB
        S1["bottleneck = model throughput,<br/>not CPU"]
        S2["min instances 1 in business hours,<br/>0 overnight"]
        S3["concurrency 8 per instance<br/><i>a turn is I/O-bound</i>"]
        S4["p95 budget 25 s;<br/>hard cap 180 s per turn"]
    end

    classDef safety fill:#7f1d1d,stroke:#fca5a5,color:#fff
    class SK safety
```

A turn is almost entirely waiting on the model, so instances carry high concurrency and the
cost driver is inference, not compute. Detail: [HLD §16](HLD.md#16-cost-scale-and-security-posture).

---

## 3. Where the data lives

Every class of data, its store, and how it is protected. Nothing personal ever reaches a
model context, a log, or a trace.

```mermaid
flowchart TB
    subgraph warehouse["Warehouse — read-only"]
        BQ[("BigQuery<br/><i>thelook_ecommerce</i>")]
        BQN["service account: bigquery.jobUser only<br/>authorized views omit denied columns<br/>dry run before every execution"]
        BQ --- BQN
    end

    subgraph knowledge["Knowledge — versioned"]
        GCS[("GCS<br/><i>golden trios · personas</i>")]
        VEC[("Vector index<br/><i>trio embeddings</i>")]
        GCSN["object versioning = rollback<br/>promoted vs candidate separated<br/>only promoted is retrievable"]
        GCS --- GCSN
        GCS --> VEC
    end

    subgraph state["Agent state — per user"]
        FS[("Firestore<br/><i>sessions · checkpoints<br/>user profiles · saved reports<br/>audit log</i>")]
        FSN["CMEK encrypted · TTL on tombstones<br/>ownership enforced in the query<br/>audit log append-only"]
        FS --- FSN
    end

    subgraph ephemeral["Ephemeral"]
        RED[("Memorystore<br/><i>semantic + schema cache</i>")]
        REDN["short TTL · masked payloads only"]
        RED --- REDN
    end

    subgraph telemetry["Telemetry"]
        TR[("Cloud Trace + Logging<br/>-> BigQuery sink")]
        TRN["prompts and SQL retained<br/>NO customer PII, ever"]
        TR --- TRN
    end

    BQ -->|"rows, masked before<br/>entering any prompt"| MASK{{"PII masker"}}
    MASK -->|"pseudonymous ids,<br/>age bands, aggregates"| CTX["model context"]
    MASK -.->|"denied columns<br/>never leave"| X["dropped"]

    classDef store fill:#1e3a8a,stroke:#93c5fd,color:#fff
    classDef safety fill:#7f1d1d,stroke:#fca5a5,color:#fff
    class BQ,GCS,VEC,FS,RED,TR store
    class MASK safety
```

The masker sits between the warehouse and the prompt, not between the model and the user —
which is what makes prompt injection unable to exfiltrate data the model never received.
Detail: [HLD §7](HLD.md#7-safety-and-pii).

---

## 4. The agent graph

Control flow inside one turn. The SQL repair cycle is real graph edges, so every attempt is
independently traced and resumable. Red nodes are gates — three deterministic, one human.

```mermaid
flowchart TD
    START([user message]) --> G{{guardrail<br/><i>deterministic, then classifier</i>}}
    G -->|blocked| RF[refusal] --> DONE([answer])
    G -->|allowed| R[retrieve precedents<br/><i>Golden Bucket, hybrid</i>]
    R --> P{{plan<br/><i>route + decompose into steps</i>}}

    P -->|schema| SC[answer from catalog] --> L
    P -->|converse| CV[answer from history] --> L
    P -->|delete_reports| RD[resolve targets<br/><i>READ ONLY</i>]
    P -->|analysis / report| GS[generate SQL]

    GS --> V{{validate SQL<br/><i>non-LLM policy gate</i>}}
    V -->|rejected| RP[repair]
    V -->|ok| EST{{cost estimate<br/><i>BQ dry run</i>}}
    EST -->|over budget| RP
    EST -->|ok| X[execute + mask results]
    X -->|error| RP
    X -->|0 rows| RP
    X -->|ok| AD[advance step]
    RP -->|budget left| V
    RP -->|budget spent| AD
    AD -->|more steps| GS
    AD -->|all steps done| SY[synthesize answer]
    SY --> PR[persist report<br/><i>if one was requested</i>] --> L[learn<br/><i>prefs + trio candidate</i>] --> DONE

    RD -->|nothing matched| DONE
    RD -->|matched| CF{{"confirm<br/><b>graph interrupt</b><br/><i>state checkpointed,<br/>nothing mutated</i>"}}
    CF -->|approved + token valid| AP[soft delete + audit] --> DONE
    CF -->|declined / wrong phrase| CX[cancel] --> DONE

    classDef gate fill:#7f1d1d,stroke:#fca5a5,color:#fff
    class G,V,EST,CF gate
```

Detail: [HLD §3](HLD.md#3-the-agent-graph) · [resilience](HLD.md#10-resilience-and-error-handling)

---

## 5. Anatomy of a turn

End-to-end data flow for *"Why are customers in Texas underspending compared to
California?"* — including where masking happens and what is written behind.

```mermaid
sequenceDiagram
    autonumber
    participant U as Manager
    participant API as Cloud Run API
    participant G as Safety kernel
    participant GB as Golden Bucket
    participant M as Gemini
    participant V as SQL validator
    participant BQ as BigQuery
    participant FS as Firestore

    U->>API: question + session id (SSE opened)
    API->>FS: load session history + user profile
    API->>G: stage 1 — deterministic patterns
    Note over G: injection / PII-request / destructive regexes<br/>no model call, ~0 ms
    G->>M: stage 2 — scope classifier (Flash)
    M-->>G: {"decision":"allow"}

    API->>GB: hybrid retrieve (BM25 + embedding)
    GB-->>API: trio_001 (0.99) + 3 more
    Note over GB: carries the analyst's SQL AND<br/>the method notes: "split frequency × AOV"

    API->>M: plan (Flash) + precedents + capability summary
    M-->>API: 2 steps, time window, guidance

    loop each step (bounded repairs)
        API->>M: generate SQL (Flash) + schema + PII rules + precedent SQL
        M-->>API: SELECT …
        API->>V: validate
        alt violation
            V-->>API: rejected + reason
            API->>M: repair with the exact violation
        else ok
            V-->>API: normalised SQL + LIMIT + PII column map
            API->>BQ: dry run — bytes estimate
            BQ-->>API: 41 MB, under ceiling
            API->>BQ: execute (maximum_bytes_billed set)
            BQ-->>API: DataFrame
            Note over API: PII layer 2 — mask BEFORE the rows<br/>enter any prompt: id→hash, age→band,<br/>denied columns dropped
        end
    end

    API->>M: synthesize (Pro) + persona + user prefs + masked results + precedent interpretation
    M-->>API: executive answer
    Note over API: PII layer 3 — regex sweep over the answer
    API-->>U: streamed answer
    par write-behind, off the response path
        API->>FS: history, preference signals, trace
        API->>GB: candidate trio (status=candidate)
    end
```

Detail: [HLD §4](HLD.md#4-anatomy-of-a-turn-data-flow)

---

## 6. Safety enforcement points

Five layers, each assuming the ones above it have failed. Only L1 involves a model.

```mermaid
flowchart TB
    IN["manager's message"] --> L0

    subgraph L0["L0 · Infrastructure"]
        direction LR
        A1["read-only service account"] --- A2["authorized views omit<br/>denied columns entirely"] --- A3["VPC-SC perimeter"]
    end

    L0 --> L1
    subgraph L1["L1 · Input guardrail"]
        direction LR
        B1["deterministic regexes<br/><i>injection · PII request · destructive</i>"] --> B2["scope classifier<br/><i>Flash, only if stage 1 passed</i>"]
    end

    L1 -->|allowed| L2
    subgraph L2["L2 · SQL validator — no LLM"]
        direction LR
        C1["single statement"] --- C2["SELECT only,<br/>at every depth"] --- C3["table allowlist"] --- C4["denied columns<br/>in ANY projection"] --- C5["no SELECT * on users"] --- C6["LIMIT injected"]
    end

    L2 --> L3
    subgraph L3["L3 · Result masking — before the prompt"]
        direction LR
        D1["deny → column dropped"] --- D2["hash → salted SHA-256"] --- D3["generalize → age band,<br/>postal prefix"] --- D4["free-text cell sweep"]
    end

    L3 --> L4
    subgraph L4["L4 · Output scrub"]
        E1["regex sweep over the final answer:<br/>email · phone · card · SSN · IP · street"]
    end

    L4 --> OUT["answer"]

    classDef safety fill:#7f1d1d,stroke:#fca5a5,color:#fff
    class L0,L1,L2,L3,L4 safety
```

Detail: [HLD §7](HLD.md#7-safety-and-pii) · [threat model](HLD.md#threat-model)

---

## Further diagrams

The design document carries six more: the hybrid retrieval pipeline and the Golden Bucket
update loop ([§6](HLD.md#6-hybrid-intelligence--the-golden-bucket)), the destructive-op
confirmation protocol ([§8](HLD.md#8-high-stakes-oversight--destructive-operations)), the
preference confidence ramp ([§9.1](HLD.md#91-user-level)), model and provider failover
([§10.3](HLD.md#103-third-party-failures)), the evaluation pyramid
([§11.1](HLD.md#111-the-evaluation-pyramid)), and the persona hot-reload path
([§13](HLD.md#13-agility--persona-management)).
