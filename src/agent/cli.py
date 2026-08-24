"""Command-line chat interface."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .config import list_personas, load_persona, setting
from .memory import user_profile
from .obs import metrics
from .obs.tracing import Trace, find_trace, recent_traces
from .resilience.policies import breaker_states
from .services import build_services
from .session import ChatSession, TurnResult

console = Console()
app = typer.Typer(add_completion=False, help="Retail Insight Agent - executive data chat.")

BANNER = r"""
  ____      _        _ _   ___           _       _     _
 |  _ \ ___| |_ __ _(_) | |_ _|_ __  ___(_) __ _| |__ | |_
 | |_) / _ \ __/ _` | | |  | || '_ \/ __| |/ _` | '_ \| __|
 |  _ <  __/ || (_| | | |  | || | | \__ \ | (_| | | | | |_
 |_| \_\___|\__\__,_|_|_| |___|_| |_|___/_|\__, |_| |_|\__|
                                           |___/
"""

HELP = """
**Ask anything about the retail data.** Some things to try:

- `How did revenue trend over the last 12 months?`
- `Why are customers in Texas underspending compared to California?`
- `Compare Jeans against Shorts and explain the difference`
- `Why did our churn rate spike last month?`
- `What data do we have available?`
- `Create a Q1 report with insights and action items for Q2`
- `Delete the reports mentioning Jeans`

**Commands**

| command | what it does |
|---|---|
| `/help` | this message |
| `/health` | warehouse, LLM providers, knowledge base, circuit breakers |
| `/trace [id]` | span-by-span breakdown of a turn — what ran, how long, what failed |
| `/sql` | the SQL behind the last answer |
| `/metrics` | agent SLIs for this process |
| `/persona [list\\|use <id>\\|show]` | switch or inspect the report persona |
| `/reports [list\\|show <id>\\|restore <id>]` | your saved report library |
| `/audit` | audit trail of report operations |
| `/prefs [show\\|forget]` | what the agent has learned about you |
| `/golden [stats\\|candidates\\|promote <id>\\|reject <id>]` | the analyst knowledge base |
| `/feedback <up\\|down> [note]` | rate the last answer |
| `/clear` | forget the conversation (keeps learned preferences) |
| `/exit` | quit |
"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_answer(result: TurnResult) -> None:
    body: List[Any] = [Markdown(result.answer)]
    footer = _footer(result)
    if footer:
        body.append(Text(""))
        body.append(footer)
    console.print(Panel(Group(*body), border_style="cyan", padding=(1, 2),
                        title="[bold cyan]Analyst[/]", title_align="left"))


def _footer(result: TurnResult) -> Optional[Text]:
    bits: List[str] = []
    state = result.state
    steps = result.steps
    ok = sum(1 for s in steps if s.get("status") == "ok")
    if steps:
        bits.append(f"{ok}/{len(steps)} queries")
    repairs = sum(s.get("repairs", 0) for s in steps)
    if repairs:
        bits.append(f"{repairs} self-repair{'s' if repairs > 1 else ''}")
    masked = [s.get("mask_note") for s in steps if s.get("mask_note")]
    if masked:
        bits.append("PII masked")
    if state.get("precedents"):
        bits.append(f"precedent {state['precedents'][0]['trio_id']}")
    bits.append(f"{result.elapsed_ms / 1000:.1f}s")
    if result.trace:
        bits.append(f"trace {result.trace.trace_id[:8]}")
    text = Text("  ·  ".join(bits), style="dim")
    if state.get("degraded"):
        text = Text(f"degraded: {state.get('degraded_reason', '')}\n", style="yellow") + text
    return text


def confirm_deletion(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Human-in-the-loop gate. Nothing has been deleted when this is called."""
    rows = payload.get("rows", [])
    table = Table(show_header=True, header_style="bold red", box=None, padding=(0, 2))
    table.add_column("id", style="dim")
    table.add_column("created")
    table.add_column("title", style="bold")
    table.add_column("matched because", style="yellow")
    for row in rows[:25]:
        table.add_row(row.get("report_id", ""), row.get("created_at", ""),
                      row.get("title", ""), row.get("reason", ""))
    if len(rows) > 25:
        table.add_row("...", "", f"and {len(rows) - 25} more", "")

    console.print(Panel(
        Group(
            Text(f"This will delete {payload.get('matched', 0)} saved report(s). "
                 f"They are recoverable for 30 days.", style="bold red"),
            Text(""),
            table,
        ),
        border_style="red", title="[bold red]⚠  Confirm deletion[/]", title_align="left",
        padding=(1, 2),
    ))

    if payload.get("requires_phrase"):
        phrase = payload.get("confirm_phrase", "")
        console.print(f"[red]This is a bulk delete.[/] Type [bold]{phrase}[/] to proceed, "
                      f"or anything else to cancel.")
        typed = Prompt.ask("[red]confirm[/]", default="", show_default=False)
        return {"approved": typed.strip().lower() == phrase.lower(), "phrase": typed}

    answer = Prompt.ask("[red]Delete these?[/] [dim](y/N)[/]", default="n", show_default=False)
    return {"approved": answer.strip().lower() in ("y", "yes"), "phrase": ""}


def render_trace(trace: Trace) -> None:
    summary = trace.summary()
    header = Table.grid(padding=(0, 2))
    header.add_column(style="dim")
    header.add_column()
    header.add_row("trace", trace.trace_id)
    header.add_row("question", trace.user_query[:90])
    header.add_row("outcome", f"{summary['outcome']}  ·  {summary['duration_ms']:.0f} ms")
    header.add_row("llm", f"{summary['llm_calls']} calls  ·  "
                          f"{summary['llm_tokens_in']} in / {summary['llm_tokens_out']} out tokens")
    header.add_row("sql", f"{summary['sql_executions']} executions  ·  "
                          f"{summary['sql_repairs']} repairs  ·  "
                          f"{summary['bytes_billed'] / 1e6:.1f} MB billed")
    if summary["errors"]:
        header.add_row("errors", ", ".join(summary["errors"]))

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("", width=2)
    table.add_column("span")
    table.add_column("kind", style="dim")
    table.add_column("ms", justify="right")
    table.add_column("detail", overflow="fold")

    depth: Dict[str, int] = {}
    for span in trace.spans:
        depth[span.span_id] = depth.get(span.parent_span_id, -1) + 1 if span.parent_span_id else 0
        indent = "  " * depth[span.span_id]
        mark = "✓" if span.status == "ok" else "✗"
        style = "green" if span.status == "ok" else "red"
        table.add_row(Text(mark, style=style), f"{indent}{span.name}", span.kind,
                      f"{span.duration_ms:.0f}", _span_detail(span))

    console.print(Panel(Group(header, Text(""), table), border_style="magenta",
                        title="[bold magenta]Trace[/]", title_align="left", padding=(1, 2)))


def _span_detail(span) -> str:
    a = span.attributes
    interesting = ["decision", "route", "steps", "hits", "ok", "violations", "rows",
                   "error_kind", "provider", "model", "masking", "outcome", "matched",
                   "estimated_bytes", "tokens_in", "tokens_out", "learned", "patterns"]
    bits = []
    for key in interesting:
        if key in a and a[key] not in (None, "", [], {}):
            value = a[key]
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value[:3])
            bits.append(f"{key}={str(value)[:70]}")
    if span.error:
        bits.append(f"[red]{span.error[:90]}[/]")
    for event in span.events[:2]:
        bits.append(f"[yellow]{event['name']}[/]")
    return "  ".join(bits)


def render_health(svc) -> None:
    health = svc.health()
    wh = health["warehouse"]
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    status_style = "green" if wh.get("status") == "ok" else "red"
    table.add_row("warehouse", f"[{status_style}]{wh.get('status')}[/] · {wh.get('backend')}")
    if wh.get("tables"):
        table.add_row("", ", ".join(f"{k}={v:,}" for k, v in wh["tables"].items()))
    providers = health["llm_providers"]
    table.add_row("llm providers",
                  ", ".join(providers) if providers else "[red]none configured[/]")
    gb = health["golden_bucket"]
    table.add_row("golden bucket",
                  f"{gb['promoted']} promoted, {gb['candidates']} candidates · {gb['embedder']}")
    cat = health["catalog"]
    table.add_row("schema catalog", f"{len(cat['loaded'])} tables loaded"
                  + (f" · errors: {cat['errors']}" if cat["errors"] else ""))
    breakers = breaker_states()
    if breakers:
        table.add_row("circuit breakers",
                      ", ".join(f"{k}={v}" for k, v in breakers.items()))
    console.print(Panel(table, border_style="green", title="[bold green]Health[/]",
                        title_align="left", padding=(1, 2)))


def render_metrics() -> None:
    snap = metrics.snapshot()
    from .obs.metrics import derived

    counters = Table(show_header=True, header_style="bold", box=None)
    counters.add_column("counter")
    counters.add_column("value", justify="right")
    for name, value in sorted(snap["counters"].items()):
        counters.add_row(name, f"{value:g}")

    hists = Table(show_header=True, header_style="bold", box=None)
    for col in ("histogram", "n", "p50", "p95", "max"):
        hists.add_column(col, justify="right" if col != "histogram" else "left")
    for name, stats in sorted(snap["histograms"].items()):
        hists.add_row(name, str(stats["count"]), f"{stats['p50']:g}",
                      f"{stats['p95']:g}", f"{stats['max']:g}")

    ratios = Table(show_header=True, header_style="bold", box=None)
    ratios.add_column("SLI")
    ratios.add_column("value", justify="right")
    for name, value in derived().items():
        ratios.add_row(name, f"{value:.2%}" if value <= 1 else f"{value:g}")

    console.print(Panel(Group(counters, Text(""), hists, Text(""), ratios),
                        border_style="blue", title="[bold blue]Metrics[/]",
                        title_align="left", padding=(1, 2)))


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------
def handle_command(raw: str, session: ChatSession) -> bool:
    """Returns False when the loop should exit."""
    parts = raw.strip().split()
    cmd, args = parts[0].lower(), parts[1:]
    svc = session.services

    if cmd in ("/exit", "/quit", "/q"):
        return False

    if cmd == "/help":
        console.print(Panel(Markdown(HELP), border_style="dim", padding=(1, 2)))

    elif cmd == "/health":
        render_health(svc)

    elif cmd == "/metrics":
        render_metrics()

    elif cmd == "/trace":
        trace = find_trace(args[0]) if args else (session.last_result.trace if session.last_result else None)
        if trace is None:
            recent = recent_traces(8)
            if not recent:
                console.print("[dim]No traces yet — ask a question first.[/]")
            else:
                for t in recent:
                    s = t.summary()
                    console.print(f"  [dim]{t.trace_id}[/]  {s['outcome']:<8} "
                                  f"{s['duration_ms']:>7.0f}ms  {t.user_query[:60]}")
        else:
            render_trace(trace)

    elif cmd == "/sql":
        result = session.last_result
        statements = result.sql_used() if result else []
        if not statements:
            console.print("[dim]The last answer didn't run any SQL.[/]")
        for i, stmt in enumerate(statements, 1):
            step = result.steps[i - 1] if i <= len(result.steps) else {}
            console.print(Panel(
                Syntax(stmt, "sql", theme="ansi_dark", word_wrap=True),
                border_style="dim",
                title=f"[dim]step {i} — {step.get('status', '?')} · "
                      f"{step.get('rows', 0)} rows · {step.get('repairs', 0)} repairs[/]",
                title_align="left"))

    elif cmd == "/persona":
        sub = args[0] if args else "show"
        if sub == "list":
            for pid in list_personas():
                p = load_persona(pid)
                marker = "[green]●[/]" if pid == session.persona_id else " "
                console.print(f" {marker} [bold]{pid}[/] — {p.get('display_name', '')} "
                              f"[dim](v{p.get('version', '?')}, by {p.get('updated_by', '?')})[/]")
        elif sub == "use" and len(args) > 1:
            try:
                load_persona(args[1])
                session.set_persona(args[1])
                console.print(f"[green]Persona switched to[/] [bold]{args[1]}[/] "
                              f"[dim](takes effect on your next question)[/]")
            except FileNotFoundError as exc:
                console.print(f"[red]{exc}[/]")
        else:
            p = load_persona(session.persona_id)
            console.print(Panel(
                Markdown(f"**{p.get('display_name')}** (`{session.persona_id}` v{p.get('version')})\n\n"
                         f"_Last edited by {p.get('updated_by')} on {p.get('updated_at')}_\n\n"
                         f"**Tone**\n{p.get('tone', '')}\n\n**Structure**\n```\n"
                         f"{p.get('default_format', '')}\n```\n\n**Rules**\n"
                         + "\n".join(f"- {r}" for r in p.get("rules", []))),
                border_style="dim", padding=(1, 2)))
            console.print("[dim]Edit config/personas/*.yaml and the change applies on the next "
                          "question — no restart.[/]")

    elif cmd == "/reports":
        sub = args[0] if args else "list"
        if sub == "show" and len(args) > 1:
            report = svc.reports.get(args[1], session.user_id)
            if report is None:
                console.print("[red]No such report (or it isn't yours).[/]")
            else:
                console.print(Panel(Markdown(report.body_md), border_style="cyan",
                                    title=f"[bold]{report.title}[/]", title_align="left",
                                    padding=(1, 2)))
        elif sub == "restore" and len(args) > 1:
            n = svc.reports.restore([args[1]], session.user_id)
            console.print(f"[green]Restored {n} report(s).[/]" if n else "[yellow]Nothing restored.[/]")
        else:
            reports = svc.reports.list(session.user_id, include_deleted=True)
            if not reports:
                console.print("[dim]No saved reports yet. Ask me to write one.[/]")
            table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
            for col in ("id", "created", "title", "state"):
                table.add_column(col)
            for r in reports:
                table.add_row(r.report_id, r.created_at[:16].replace("T", " "), r.title,
                              "[red]deleted[/]" if r.deleted else "[green]active[/]")
            console.print(table)

    elif cmd == "/audit":
        entries = svc.reports.audit_trail(session.user_id, limit=25)
        if not entries:
            console.print("[dim]No report operations recorded yet.[/]")
        for e in entries:
            console.print(f"  [dim]{e['ts']}[/]  [bold]{e['action']:<26}[/] "
                          f"{len(e['targets'])} target(s)  [dim]{str(e['detail'])[:80]}[/]")

    elif cmd == "/prefs":
        if args and args[0] == "forget":
            key = args[1] if len(args) > 1 else None
            n = user_profile.forget(session.user_id, key)
            console.print(f"[green]Forgot {n} preference(s).[/]")
        else:
            prefs = user_profile.get_preferences(session.user_id)
            if not prefs:
                console.print("[dim]I haven't learned anything about you yet. Tell me how you "
                              "like your answers and I'll remember.[/]")
            table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
            for col in ("preference", "value", "source", "confidence", "seen", "active", "from"):
                table.add_column(col)
            for p in prefs:
                table.add_row(p.key, p.value, p.source, f"{p.confidence:.2f}",
                              str(p.evidence_count),
                              "[green]yes[/]" if p.active else "[dim]not yet[/]",
                              f"[dim]{p.last_evidence[:40]}[/]")
            console.print(table)

    elif cmd == "/golden":
        sub = args[0] if args else "stats"
        if sub == "candidates":
            cands = svc.golden.list_candidates()
            if not cands:
                console.print("[dim]No candidate trios awaiting review.[/]")
            for c in cands:
                console.print(f"  [bold]{c.trio_id}[/] [dim]quality {c.quality_score:.2f}[/] "
                              f"{c.question[:70]}")
                console.print(f"     [dim]tags: {', '.join(c.intent_tags)}[/]")
                for note in c.analyst_method_notes:
                    console.print(f"     [dim]· {note[:100]}[/]")
        elif sub == "promote" and len(args) > 1:
            path = svc.golden.promote(args[1], reviewer=session.user_id)
            console.print(f"[green]Promoted → {path}[/]" if path else "[red]No such candidate.[/]")
        elif sub == "reject" and len(args) > 1:
            console.print("[green]Rejected.[/]" if svc.golden.reject(args[1]) else "[red]No such candidate.[/]")
        else:
            stats = svc.golden.stats()
            console.print(f"  promoted trios : [bold]{stats['promoted']}[/]")
            console.print(f"  candidates     : [bold]{stats['candidates']}[/] "
                          f"[dim](review with /golden candidates)[/]")
            console.print(f"  embedder       : {stats['embedder']}")
            console.print(f"  intent tags    : [dim]{', '.join(stats['tags'][:14])}[/]")

    elif cmd == "/feedback":
        if not session.last_result:
            console.print("[dim]Ask something first.[/]")
        elif not args or args[0] not in ("up", "down"):
            console.print("[dim]Usage: /feedback up|down [note][/]")
        else:
            user_profile.record_feedback(
                session.user_id, session.session_id,
                session.last_result.trace.trace_id if session.last_result.trace else "",
                args[0], " ".join(args[1:]),
                session.history[-2]["content"] if len(session.history) >= 2 else "",
                session.last_result.answer,
            )
            console.print("[green]Noted — that feeds the offline eval set.[/]")

    elif cmd == "/clear":
        session.history.clear()
        console.print("[green]Conversation cleared.[/] [dim]Learned preferences kept "
                      "(use /prefs forget to reset those).[/]")

    elif cmd == "/whoami":
        console.print(f"  user: [bold]{session.user_id}[/]  session: {session.session_id}  "
                      f"persona: {session.persona_id}")

    else:
        console.print(f"[yellow]Unknown command {cmd}. Try /help.[/]")

    return True


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
@app.command()
def chat(
    user: str = typer.Option("manager_a", "--user", "-u", help="Manager identity (scopes reports and learned preferences)."),
    name: str = typer.Option("", "--name", help="Display name used in reports."),
    persona: str = typer.Option("", "--persona", "-p", help="Persona id (default from settings.yaml)."),
    warehouse: str = typer.Option("", "--warehouse", "-w", help="offline | bigquery"),
    ask: Optional[str] = typer.Option(None, "--ask", help="Ask one question and exit."),
) -> None:
    console.print(Text(BANNER, style="bold cyan"))
    persona_id = persona or setting("runtime.default_persona", "exec_default")

    with console.status("[dim]starting up — attaching warehouse and knowledge base...[/]"):
        try:
            svc = build_services(warehouse or None)
        except Exception as exc:
            console.print(Panel(f"[red]Could not start:[/] {exc}\n\n"
                                "See docs/SETUP.md — the offline mode needs no credentials at all.",
                                border_style="red"))
            raise typer.Exit(1)

    session = ChatSession(services=svc, user_id=user, persona_id=persona_id,
                          user_display_name=name or user)

    render_health(svc)
    if not svc.router.available_providers():
        console.print(Panel(
            "[yellow]No LLM provider is configured, so I can't reason about the data yet.[/]\n\n"
            "Set one of these and restart:\n"
            "  export GOOGLE_API_KEY=...        [dim]# free key: https://aistudio.google.com/apikey[/]\n"
            "  export OPENROUTER_API_KEY=...\n"
            "  export OLLAMA_ENABLED=1\n\n"
            "[dim]Everything else — the warehouse, the safety layer, the report library — is live.[/]",
            border_style="yellow", title="[yellow]Setup needed[/]", title_align="left"))

    if ask:
        _run_turn(session, ask)
        return

    console.print(f"\n[dim]Signed in as[/] [bold]{user}[/] [dim]· persona[/] [bold]{persona_id}[/]"
                  f"[dim] · type /help for commands, /exit to quit[/]\n")

    while True:
        try:
            message = Prompt.ask(f"[bold green]{user}[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/]")
            break
        if not message:
            continue
        if message.startswith("/"):
            if not handle_command(message, session):
                console.print("[dim]Bye.[/]")
                break
            continue
        _run_turn(session, message)


def _run_turn(session: ChatSession, message: str) -> None:
    console.print()
    started = time.perf_counter()
    with console.status("[dim]thinking...[/]", spinner="dots"):
        result = session.ask(message, on_confirm=confirm_deletion)
    render_answer(result)
    console.print()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
