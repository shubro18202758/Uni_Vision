"""End-to-end smoke test: Gemma 4 E2B via AgentLLMClient."""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def main():
    from uni_vision.agent.llm_client import AgentLLMClient
    from uni_vision.common.config import OllamaConfig

    cfg = OllamaConfig()
    print(f"Model: {cfg.model}")
    print(f"Base URL: {cfg.base_url}")

    client = AgentLLMClient(config=cfg, timeout_s=60.0, max_tokens=512)

    # ── Test 1: Simple Q&A ───────────────────────────────────────
    resp = await client.chat(
        [
            {"role": "system", "content": "You are a helpful assistant. Reply concisely."},
            {"role": "user", "content": "What is ANPR in one sentence?"},
        ],
        temperature=0.2,
    )
    print("\n--- Test 1: Simple Q&A ---")
    print(f"Content: {resp.content[:300]}")
    print(f"Eval count: {resp.eval_count}")
    assert resp.content, "FAIL: empty content"
    print("PASS")

    # ── Test 2: ReAct tool-call format ───────────────────────────
    system_prompt = (
        "You are an ANPR pipeline agent. Available tools:\n"
        "- get_system_health: Check pipeline health status\n"
        "- query_detections: Query recent plate detections\n\n"
        'Respond ONLY in JSON. To use a tool: {"thought": "...", "action": "tool_name", "arguments": {}}\n'
        'To give a final answer: {"thought": "...", "answer": "..."}'
    )

    resp2 = await client.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Is the system healthy?"},
        ],
        temperature=0.2,
    )
    print("\n--- Test 2: ReAct Tool Call ---")
    print(f"Content: {resp2.content}")
    # Verify it's valid JSON with an action
    parsed = json.loads(resp2.content.strip().strip("`").strip())
    assert "action" in parsed or "answer" in parsed, "FAIL: no action or answer key"
    print("PASS")

    # ── Test 3: Multi-turn with observation ──────────────────────
    resp3 = await client.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Is the system healthy?"},
            {
                "role": "assistant",
                "content": '{"thought": "I need to check system health.", "action": "get_system_health", "arguments": {}}',
            },
            {
                "role": "user",
                "content": 'Observation: {"pipeline_running": true, "fps": 24.5, "gpu_util": 0.65, "queue_depth": 3}',
            },
        ],
        temperature=0.2,
    )
    print("\n--- Test 3: Post-Observation Answer ---")
    print(f"Content: {resp3.content}")
    parsed3 = json.loads(resp3.content.strip().strip("`").strip())
    assert "answer" in parsed3, "FAIL: expected final answer after observation"
    print("PASS")

    await client.close()
    print("\n" + "=" * 50)
    print("ALL SMOKE TESTS PASSED — Gemma 4 E2B is operational")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
