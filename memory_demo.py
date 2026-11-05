"""Task 8 transcript: state carried within one session, absent in a fresh one."""
from agent_service import ask_agent, history_of

print("=== Session A (multi-turn) ===")
for turn in ["What is the return window for footwear?", "And what about Beauty?",
             "Check order ORD0039 please", "Is it delayed?"]:
    r = ask_agent(turn, session_id="session-A")
    print(f"user: {turn}   [resolved: {r['resolved_query']}]\n  agent: {r['answer'][:160]}\n  sources: {r['sources']}")
print("session-A history:", len(history_of("session-A")), "messages")

print("\n=== Session B (fresh conversation, same follow-up words) ===")
for turn in ["And what about Beauty?", "Is it delayed?"]:
    r = ask_agent(turn, session_id="session-B")
    print(f"user: {turn}   [resolved: {r['resolved_query']}]\n  agent: {r['answer'][:160]}\n  sources: {r['sources']}")
print("session-B history:", len(history_of("session-B")), "messages (no footwear, no ORD0039)")
