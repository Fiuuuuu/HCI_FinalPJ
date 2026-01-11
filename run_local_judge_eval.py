"""Local LLM-as-a-Judge evaluation runner (no Langfuse required).

- Reads questions from a Markdown question bank (e.g. eval_question_bank_v1.md).
- Runs the system-under-test (SUT) using Agents SDK directly (triage_agent from agents_demo).
- Uses an LLM judge (OpenAI-compatible, e.g. Qwen DashScope) to score helpfulness.
- Writes results to JSONL (and optional CSV).

Usage:
  python run_local_judge_eval.py --questions-file eval_question_bank_v1.md --out results.jsonl

Optional:
  python run_local_judge_eval.py --limit 20 --csv results.csv

Environment (preferred):
  JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL

If not provided, falls back to agents_demo.main_qwen BASE_URL/API_KEY/MODEL_NAME1.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
AGENTS_DEMO_SRC = ROOT / "agents_demo 4" / "src"
if AGENTS_DEMO_SRC.exists():
    sys.path.insert(0, str(AGENTS_DEMO_SRC))
else:
    raise SystemExit(f"Cannot find agents_demo src at: {AGENTS_DEMO_SRC}")

# Load env vars from common locations (optional)
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "agents_demo 4" / ".env")
except Exception:
    pass

# Keep Agents SDK tracing quiet by default (can retry/time out if no collector is reachable).
# If you want richer tracing, run with `--agents-tracing`.
os.environ.setdefault("AGENTS_TRACING_DISABLED", "1")


@dataclass(frozen=True)
class QAItem:
    qid: str
    question: str


QUESTION_LINE_RE = re.compile(r"^\s*-\s+#(?P<id>[A-Za-z0-9_-]+)\s+(?P<q>.+?)\s*$")


def load_questions_from_md(path: Path) -> list[QAItem]:
    if not path.exists():
        raise FileNotFoundError(path)

    items: list[QAItem] = []
    in_section_a = False

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "## A. 只用于跑系统的提问列表（可直接复制粘贴）":
            in_section_a = True
            continue
        if in_section_a and line.startswith("## B."):
            break

        if not in_section_a:
            continue

        m = QUESTION_LINE_RE.match(line)
        if m:
            qid = m.group("id").strip()
            q = m.group("q").strip()
            items.append(QAItem(qid=qid, question=q))

    if not items:
        raise ValueError(
            "No questions found. Expected '- #ID question' lines under section A in the markdown file."
        )
    return items


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


async def run_sut(question: str) -> dict[str, Any]:
    # Import lazily so the script can start even if Agents SDK import is slow.
    from agents import Runner
    from agents.items import HandoffOutputItem, MessageOutputItem, ItemHelpers
    from agents_demo.main_qwen import create_initial_context, triage_agent, myRunConfig, OpenAIModel

    ctx = create_initial_context()

    if OpenAIModel:
        result = await Runner.run(triage_agent, question, context=ctx)
    else:
        result = await Runner.run(triage_agent, question, context=ctx, run_config=myRunConfig)

    messages: list[dict[str, Any]] = []
    agent_trajectory: list[str] = []

    for item in getattr(result, "new_items", []) or []:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            agent_name = getattr(getattr(item, "agent", None), "name", None) or "unknown"
            messages.append({"agent": agent_name, "text": text})
            if not agent_trajectory or agent_trajectory[-1] != agent_name:
                agent_trajectory.append(agent_name)
        elif isinstance(item, HandoffOutputItem):
            from_agent = getattr(getattr(item, "from_agent", None), "name", None) or "unknown"
            to_agent = getattr(getattr(item, "to_agent", None), "name", None) or "unknown"
            agent_trajectory.append(f"HANDOFF:{from_agent}->{to_agent}")

    answer = (getattr(result, "final_output", None) or "").strip()
    final_agent = messages[-1]["agent"] if messages else None

    return {
        "answer": answer,
        "final_agent": final_agent,
        "agent_trajectory": agent_trajectory,
        "messages": messages,
        "raw_final_output": getattr(result, "final_output", None),
    }


def _langfuse_enabled() -> bool:
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and (os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL"))
    )


def _get_langfuse_client():
    from langfuse import get_client

    return get_client()


JUDGE_PROMPT_ZH = """你是一个严格但公平的用户体验评测员（LLM-as-a-Judge）。

请基于“普通旅客咨询航空客服”的场景，对助手回答进行【有用性/帮助程度(Helpfulness)】打分。

评分标准（1-5）：
1 = 完全无用/答非所问/明显拒答或敷衍
2 = 有一点相关但大部分无用或缺关键步骤
3 = 基本可用但不够具体或有明显遗漏
4 = 很有帮助，步骤清晰，能指导下一步
5 = 非常有帮助，准确、清晰、覆盖关键边界情况且不胡编

重要约束：
- 如果回答拒绝或说“只能回答某类问题”等，通常 ≤2。
- 如果回答编造具体金额/政策且没有依据，扣分。
- 如果回答提供明确的下一步（需要哪些信息、去哪里操作、注意事项），加分。

只输出 JSON，不要输出任何其他文字。
输出格式：
{
  "score": 1-5,
  "reason": "一句到三句中文理由",
  "issues": ["问题点1", "问题点2"],
  "highlights": ["优点1", "优点2"]
}
"""


async def judge_helpfulness(question: str, answer: str, *, base_url: str, api_key: str, model: str) -> dict[str, Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    messages = [
        {"role": "system", "content": JUDGE_PROMPT_ZH},
        {
            "role": "user",
            "content": f"用户问题：{question}\n\n助手回答：{answer}",
        },
    ]

    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"Judge did not return valid JSON. Raw content: {content}") from e

    # Minimal normalization
    score = data.get("score")
    if isinstance(score, str) and score.isdigit():
        data["score"] = int(score)
    return data


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # Flatten a few known fields for convenience
    fieldnames = [
        "qid",
        "question",
        "answer",
        "judge_score",
        "judge_reason",
        "ts_utc",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "qid": r.get("qid"),
                    "question": r.get("question"),
                    "answer": r.get("answer"),
                    "judge_score": (r.get("judge") or {}).get("score"),
                    "judge_reason": (r.get("judge") or {}).get("reason"),
                    "ts_utc": r.get("ts_utc"),
                }
            )


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions-file",
        default=str(ROOT / "eval_question_bank_v1.md"),
        help="Markdown question bank file (default: eval_question_bank_v1.md)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Only evaluate first N questions (0 = all)")
    ap.add_argument("--out", default=str(ROOT / "local_eval_results.jsonl"), help="Output JSONL path")
    ap.add_argument("--csv", default="", help="Optional CSV output path")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between items (rate limiting)")

    ap.add_argument(
        "--langfuse",
        action="store_true",
        help="Write traces + judge scores to Langfuse (requires LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST|BASE_URL)",
    )
    ap.add_argument(
        "--agents-tracing",
        action="store_true",
        help="Enable Agents SDK tracing (default is disabled to avoid OTEL timeouts)",
    )
    ap.add_argument(
        "--langfuse-session",
        default="",
        help="Optional Langfuse session_id for grouping runs (default: autogenerated)",
    )

    ap.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL", ""))
    ap.add_argument("--judge-api-key", default=os.getenv("JUDGE_API_KEY", ""))
    ap.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", ""))

    args = ap.parse_args()

    if args.agents_tracing:
        os.environ["AGENTS_TRACING_DISABLED"] = "0"

    q_path = Path(args.questions_file)
    questions = load_questions_from_md(q_path)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    # Judge config fallback to agents_demo.main_qwen
    if not args.judge_base_url or not args.judge_api_key or not args.judge_model:
        from agents_demo.main_qwen import BASE_URL as FALLBACK_BASE_URL, API_KEY as FALLBACK_API_KEY, MODEL_NAME1 as FALLBACK_MODEL

        base_url = args.judge_base_url or FALLBACK_BASE_URL
        api_key = args.judge_api_key or FALLBACK_API_KEY
        model = args.judge_model or FALLBACK_MODEL
    else:
        base_url = args.judge_base_url
        api_key = args.judge_api_key
        model = args.judge_model

    results: list[dict[str, Any]] = []

    langfuse = None
    session_id = args.langfuse_session.strip() or None
    if args.langfuse:
        if not _langfuse_enabled():
            raise SystemExit(
                "Langfuse not configured. Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST (or LANGFUSE_BASE_URL)."
            )
        langfuse = _get_langfuse_client()
        try:
            ok = langfuse.auth_check()
        except Exception as e:
            raise SystemExit(f"Langfuse auth_check failed: {e}")
        if not ok:
            raise SystemExit("Langfuse auth_check returned False. Check keys/host.")
        if session_id is None:
            session_id = langfuse.create_trace_id(seed="local-eval-session")

    for idx, item in enumerate(questions, start=1):
        print(f"[{idx}/{len(questions)}] {item.qid} {item.question}")

        trace_id = None
        if langfuse is not None:
            trace_id = langfuse.create_trace_id(seed=f"local-eval:{session_id}:{item.qid}:{idx}")

        # If Langfuse is enabled, create a trace for this item using observe() context.
        if langfuse is not None:
            from langfuse import observe

            @observe(name="local.eval", as_type="chain")
            async def _run_one():
                # 执行SUT
                sut_local = await run_sut(item.question)
                answer_local = sut_local["answer"]
                
                # 执行Judge
                judge_local = await judge_helpfulness(
                    item.question,
                    answer_local,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                )

                # 更新trace，包含所有详细信息
                # 精简messages避免metadata过大
                messages_summary = []
                for i, msg in enumerate(sut_local.get("messages", [])[:10]):  # 只保留前10条
                    messages_summary.append({
                        "idx": i+1,
                        "agent": msg.get("agent"),
                        "text_preview": msg.get("text", "")[:200]  # 截断到200字符
                    })
                
                langfuse.update_current_trace(
                    name="local.eval",
                    session_id=session_id,
                    input={"qid": item.qid, "question": item.question},
                    output={
                        "answer": answer_local,
                        "final_agent": sut_local.get("final_agent"),
                        "score": judge_local.get("score"),
                    },
                    metadata={
                        "qid": item.qid,
                        "index": idx,
                        "judge_model": model,
                        "trace_id": trace_id,
                        # SUT详细信息（精简版）
                        "sut_final_agent": sut_local.get("final_agent"),
                        "sut_agent_trajectory": sut_local.get("agent_trajectory"),
                        "sut_message_count": len(sut_local.get("messages", [])),
                        "sut_messages_preview": messages_summary,
                        "sut_answer_length": len(answer_local),
                        # Judge详细信息
                        "judge": {
                            "score": judge_local.get("score"),
                            "reason": judge_local.get("reason"),
                            "issues": judge_local.get("issues", []),
                            "highlights": judge_local.get("highlights", []),
                        },
                    },
                )

                score = judge_local.get("score")
                try:
                    score_value = float(score) if score is not None else None
                except Exception:
                    score_value = None

                if score_value is not None:
                    langfuse.create_score(
                        name="helpfulness_local_judge",
                        value=score_value,
                        trace_id=trace_id,
                        data_type="NUMERIC",
                        comment=str(judge_local.get("reason") or ""),
                    )
                langfuse.flush()
                return sut_local, judge_local

            sut, judge = await _run_one()
            answer = sut["answer"]
        else:
            sut = await run_sut(item.question)
            answer = sut["answer"]
            judge = await judge_helpfulness(
                item.question,
                answer,
                base_url=base_url,
                api_key=api_key,
                model=model,
            )

        row = {
            "qid": item.qid,
            "question": item.question,
            "answer": answer,
            "final_agent": sut.get("final_agent"),
            "agent_trajectory": sut.get("agent_trajectory"),
            "messages": sut.get("messages"),
            "judge": judge,
            "ts_utc": _utc_iso(),
            "judge_model": model,
            "langfuse_trace_id": trace_id,
        }
        results.append(row)

        if args.sleep and args.sleep > 0:
            await asyncio.sleep(args.sleep)

    out_path = Path(args.out)
    write_jsonl(out_path, results)

    if args.csv:
        write_csv(Path(args.csv), results)

    print(f"\nWrote {len(results)} rows to: {out_path}")
    if args.csv:
        print(f"Wrote CSV to: {args.csv}")

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
