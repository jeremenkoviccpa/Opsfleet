# Architecture Diagram

The system at a glance. Each diagram below answers one of the questions the brief asks:
**what services, what talks to what, what compute runs where, and where the data lives.**

The reasoning behind every choice — why LangGraph, why Gemini tiered, why BigQuery
`VECTOR_SEARCH` before Vertex AI Vector Search — is in
**[the design document](HLD.md)**, which also carries six further diagrams covering
retrieval, the learning loops, failover and the evaluation pyramid.

| # | Diagram | Question it answers | PNG |
|---|---|---|---|
| 1 | [System architecture](#1-system-architecture) | What services, and how do they communicate? | [png](diagrams/1-system-architecture.png) |
| 2 | [Compute and request path](#2-compute-and-request-path) | What compute runs where, and how does it scale? | [png](diagrams/2-compute-and-request-path.png) |
| 3 | [Where the data lives](#3-where-the-data-lives) | How and where is data stored and handled? | [png](diagrams/3-where-the-data-lives.png) |
| 4 | [The agent graph](#4-the-agent-graph) | What is the control flow inside a turn? | [png](diagrams/4-agent-graph.png) |
| 5 | [Anatomy of a turn](#5-anatomy-of-a-turn) | What is the end-to-end data flow? | [png](diagrams/5-anatomy-of-a-turn.png) |
| 6 | [Safety enforcement points](#6-safety-enforcement-points) | Where is policy actually enforced? | [png](diagrams/6-safety-enforcement-points.png) |

> **Viewing these.** The diagrams are Mermaid, which **GitHub renders automatically** —
> if you are reading this on github.com you are seeing pictures, not code. Viewing the
> raw file in an editor shows the Mermaid source instead; VS Code needs the
> *Markdown Preview Mermaid Support* extension to render it locally. Rendered PNGs are
> committed under [`docs/diagrams/`](diagrams) for offline viewing, and
> `make diagrams` regenerates them from the Mermaid source.

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
        FLASH["Gemini Flash<br/><i>guardrail · plan · SQL · repair</i>"]
        PRO["Gemini Pro<br/><i>analysis · reports · judge</i>"]
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
flowchart TB
    U(["manager's browser / CLI"]) -->|"HTTPS + SSE<br/><i>tokens stream as they generate</i>"| FA

    subgraph CR["Cloud Run — one container, autoscaled 1..N"]
        direction TB
        FA["<b>FastAPI gateway</b><br/><i>auth · per-user quota · rate limit</i>"]
        FA --> LG["<b>LangGraph orchestrator</b><br/><i>in-process, ~200 MB resident</i>"]
        LG --> SK["<b>Safety kernel</b><br/><i>pure CPU, no I/O, sub-millisecond</i>"]
        LG --> IO["<b>I/O fan-out</b><br/><i>models · warehouse · state</i>"]
    end

    IO --> EXT{{"the turn is ~95% waiting<br/>on these"}}
    EXT --> M1["Vertex AI<br/><i>6-8 calls/turn</i>"]
    EXT --> M2["BigQuery<br/><i>1-4 queries/turn</i>"]
    EXT --> M3["Firestore<br/><i>history · checkpoints</i>"]

    IO -.->|"fire and forget"| PS["Pub/Sub"] --> CFN["Cloud Functions<br/><i>trio indexing · config validation</i>"]
    FA -.->|"if > 60 s of work"| CT["Cloud Tasks"] --> LRP["Long reports<br/><i>4+ queries, Pro model</i>"]
    SCH["Cloud Scheduler"] --> NB["Nightly evals<br/>+ trio quality decay"]

    SCALE["<b>Scaling</b> — bottleneck is model throughput, not CPU, so instances carry<br/>concurrency 8. Min 1 instance in business hours, 0 overnight.<br/>p95 budget 25 s · hard cap 180 s per turn."]
    CR -.- SCALE

    classDef safety fill:#7f1d1d,stroke:#fca5a5,color:#fff
    classDef note fill:#1f2937,stroke:#6b7280,color:#e5e7eb
    class SK safety
    class SCALE,EXT note
```

A turn is almost entirely waiting on the model, so instances carry high concurrency and the
cost driver is inference, not compute. Detail: [HLD §16](HLD.md#16-cost-scale-and-security-posture).

---

## 3. Where the data lives

Every class of data, its store, and how it is protected. Nothing personal ever reaches a
model context, a log, or a trace.

```mermaid
flowchart TD
    BQ[("<b>BigQuery</b><br/>thelook_ecommerce<br/><i>raw transaction logs, contains PII</i>")]
    BQ --> GATE

    subgraph GATE["The masking boundary — nothing personal crosses it"]
        direction LR
        M1["<b>deny</b><br/>name · address<br/>coordinates"] --> D1(["dropped —<br/>never leaves"])
        M2["<b>hash</b><br/>user id · email"] --> D2(["cust_a41f9c"])
        M3["<b>generalize</b><br/>age · postcode"] --> D3(["30-39 · 945**"])
        M4["<b>allow</b><br/>state · category<br/>aggregates"] --> D4(["verbatim"])
    end

    GATE --> CTX["<b>Model context</b><br/><i>this is the only shape<br/>the LLM ever receives</i>"]
    CTX --> ANS["Answer<br/><i>+ regex scrub</i>"]

    BQACC["<b>Access:</b> service account with bigquery.jobUser only<br/>authorized views omit denied columns · dry run before every execution"]
    BQ -.- BQACC

    subgraph STORES["Everything else the system persists"]
        direction LR
        GCS[("<b>GCS</b><br/>golden trios · personas<br/><i>object versioning = rollback</i>")]
        VEC[("<b>Vector index</b><br/>trio embeddings<br/><i>only promoted trios</i>")]
        FS[("<b>Firestore</b><br/>sessions · checkpoints · profiles<br/>saved reports · audit log<br/><i>CMEK · TTL on tombstones</i>")]
        RED[("<b>Memorystore</b><br/>semantic + schema cache<br/><i>short TTL · masked payloads only</i>")]
        TR[("<b>Cloud Trace + Logging</b><br/>prompts · SQL · decisions<br/><i>never customer PII</i>")]
        GCS --> VEC
    end

    classDef store fill:#1e3a8a,stroke:#93c5fd,color:#fff
    classDef safety fill:#7f1d1d,stroke:#fca5a5,color:#fff
    classDef out fill:#14532d,stroke:#86efac,color:#fff
    class BQ,GCS,VEC,FS,RED,TR store
    class GATE,M1,M2,M3,M4 safety
    class D1,D2,D3,D4,CTX,ANS out
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
