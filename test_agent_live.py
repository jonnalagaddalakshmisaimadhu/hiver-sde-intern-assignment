"""
Live Interactive Support Agent Tester.
Test custom customer messages directly against the @AppleSupport AI Agent.

Usage:
  python test_agent_live.py                     # Starts interactive chat prompt
  python test_agent_live.py "Your message here" # Directly tests a specific inquiry
"""
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 stdout if supported
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.agent.agent import AppleSupportAgent


def format_agent_response(query: str, resp, latency_ms: float):
    print("\n" + "=" * 80)
    print(f"CUSTOMER INQUIRY : \"{query}\"")
    print("-" * 80)
    print(f"PREDICTED INTENT : {resp.intent} (Confidence: {resp.intent_confidence:.2f})")
    
    badge = "[AUTO_HANDLE]" if resp.decision == "AUTO_HANDLE" else "[ESCALATE TO HUMAN]"
    print(f"ACTION DECISION  : {badge}")
    print(f"DECISION REASON  : {resp.reason}")
    print(f"DRAFTED REPLY    : \"{resp.reply}\"")
    print(f"LATENCY          : {latency_ms:.1f} ms")
    
    if resp.evidence:
        top_sim = resp.evidence[0].similarity
        print(f"\nRETRIEVED HISTORICAL EVIDENCE ({len(resp.evidence)} matches, Top Sim: {top_sim:.3f}):")
        for i, ev in enumerate(resp.evidence, 1):
            print(f"  [{i}] (Sim: {ev.similarity:.3f})")
            print(f"      Historical Customer: \"{ev.customer_message}\"")
            print(f"      Historical Reply   : \"{ev.brand_response}\"")
    print("=" * 80 + "\n")


def run_interactive(agent: AppleSupportAgent):
    print("\n" + "=" * 80)
    print("      @AppleSupport AI Support Agent - Live Interactive Testing")
    print("=" * 80)
    print("Type any customer inquiry below to test the agent live.")
    print("Type 'exit' or 'q' to quit.\n")

    while True:
        try:
            query = input("Inquiry > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "q", "quit"]:
                print("\nExiting live testing. Goodbye!")
                break

            t0 = time.time()
            resp = agent.process_inquiry(query)
            t_elapsed = (time.time() - t0) * 1000

            format_agent_response(query, resp, t_elapsed)

        except KeyboardInterrupt:
            print("\nExiting live testing. Goodbye!")
            break


def main():
    # If user provided a query as command line argument, run that directly
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        query = " ".join(sys.argv[1:])
        print("[INFO] Initializing @AppleSupport Agent...")
        agent = AppleSupportAgent()
        t0 = time.time()
        resp = agent.process_inquiry(query)
        t_elapsed = (time.time() - t0) * 1000
        format_agent_response(query, resp, t_elapsed)
        return

    # Otherwise default directly to interactive prompt
    print("[INFO] Initializing @AppleSupport Agent...")
    agent = AppleSupportAgent()
    run_interactive(agent)


if __name__ == "__main__":
    main()
