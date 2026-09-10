import os
import sys
import time
import uuid

from openai import OpenAI

# ==========================================
# Configuration (Set your credentials here)
# ==========================================
API_KEY = os.environ.get("LLMRIO_API_KEY")
BASE_URL = (
    os.environ.get("LLMRIO_API_URL", "http://127.0.0.1:8002").rstrip("/").removesuffix("/v1")
    + "/v1"
)
MODEL_ID = "qwen3.8-27b-nvfp4-ext"  # Replace with your loaded model ID

# Long-context test parameters
# This is only a character-based estimate. The server's reported prompt_tokens
# after the request is the authoritative token count.
NATIVE_CONTEXT_TOKENS = 260_000
TARGET_TOKENS = 600_000
CHARS_PER_TOKEN = 4
TOTAL_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN

# Placement depth: 0.0 = top, 0.5 = middle, 1.0 = bottom
# At 50%, the final question must retrieve information from the middle of a
# long causal context. Run separate trials at other depths for broader testing.
NEEDLE_DEPTH = 0.50


def generate_haystack_with_needle(target_chars: int, needle_depth: float):
    """
    Generates filler text and hides a unique secret needle at the specified depth.
    """
    secret_code = f"YARN-VERIFY-{uuid.uuid4().hex[:8].upper()}"
    needle = f"\n\n[CRITICAL NOTE: The special activation code is '{secret_code}'.]\n\n"

    # Filler sentence (~100 chars)
    filler = (
        "The quick brown fox jumps over the lazy dog repeatedly across "
        "the vast open computational plains. "
    )

    # Calculate how many repetitions are needed
    remaining_chars = max(0, target_chars - len(needle))
    num_repeats = remaining_chars // len(filler)

    split_index = int(num_repeats * needle_depth)

    # Build text halves
    first_half = filler * split_index
    second_half = filler * (num_repeats - split_index)

    context = first_half + needle + second_half
    return context, secret_code


def format_token_count(value):
    """Format an optional token count returned by an OpenAI-compatible server."""
    return f"{value:,}" if isinstance(value, int) else "not reported"


def run_yarn_context_test():
    print("=" * 60)
    print(" 1M Context Window / YaRN Implementation Test")
    print("=" * 60)
    print(f"Endpoint : {BASE_URL}")
    print(f"Model ID : {MODEL_ID}")
    print(f"Target   : ~{TARGET_TOKENS:,} tokens (~{TOTAL_CHARS:,} chars)")
    print(f"Depth    : {int(NEEDLE_DEPTH * 100)}% of total context\n")

    if not API_KEY:
        raise SystemExit("Set LLMRIO_API_KEY before running the YaRN demo")

    # 1. Initialize Client
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=600.0,  # Generous timeout for processing large prefill
    )

    # 2. Construct Payload
    print("[1/3] Generating 1M token context payload in memory...")
    context_text, expected_code = generate_haystack_with_needle(TOTAL_CHARS, NEEDLE_DEPTH)

    prompt = (
        f"{context_text}\n\n"
        "Question: What is the special activation code mentioned in the text? "
        "Return ONLY the exact code."
    )

    messages = [{"role": "user", "content": prompt}]

    # 3. Send Request
    print("[2/3] Sending request to endpoint (Prefill on 1M tokens may take some time)...")
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL_ID, messages=messages, temperature=0.0, max_tokens=50
        )
        elapsed_time = time.time() - start_time

        reply = response.choices[0].message.content.strip()
        usage = response.usage

        print(f"[3/3] Response received in {elapsed_time:.2f}s\n")

        # 4. Verification & Output
        print("-" * 40)
        print("Test Results:")
        print("-" * 40)
        if usage:
            print(f"Reported Prompt Tokens     : {usage.prompt_tokens:,}")
            print(f"Reported Completion Tokens : {usage.completion_tokens}")
            print(f"Reported Total Tokens      : {usage.total_tokens:,}")

        print(f"\nExpected Code : {expected_code}")
        print(f"Model Output  : {reply}")
        print("-" * 40)

        # Verification check
        if expected_code in reply:
            print("✅ SUCCESS: Model successfully retrieved the needle beyond the 260k boundary!")
            print("   YaRN long-context scaling is functioning properly.")
        else:
            print("❌ FAILURE: Needle was not retrieved accurately.")
            print("   Context may have been truncated, corrupted, or attention degraded.")

    except Exception as e:
        print(f"\n❌ ERROR during API call: {e}")
        print(
            "Note: If you received a context limit or out-of-memory error, "
            "verify server-side YaRN configs."
        )
        sys.exit(1)


if __name__ == "__main__":
    run_yarn_context_test()
