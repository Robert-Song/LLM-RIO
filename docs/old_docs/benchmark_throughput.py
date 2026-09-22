import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Optional
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_URL = "http://localhost:8002/v1"
API_KEY = "rio_43e22ad6dc6d_0ZKev_Uydo3GsuC2hzhzjA3UQk54FsdvTBA7D-cfois"  # vLLM accepts any string if no auth is configured
MODEL_NAME = "qwen3.8-27b-nvfp4"

CONCURRENCY = 128          # Number of concurrent workers (batching slots)
NUM_PROMPTS = 256          # Total requests to send across the run
MAX_TOKENS = 512          # Generation target length per request
TEMPERATURE = 0.7

PROMPT_TEXT = (
    "Write a detailed technical analysis comparing asynchronous event loops "
    "with multi-threaded execution models in systems programming."
)

# -----------------------------------------------------------------------------
# Data Tracking
# -----------------------------------------------------------------------------
@dataclass
class RequestResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_sec: float = 0.0
    ttft_sec: float = 0.0
    itl_list: List[float] = field(default_factory=list)
    success: bool = True
    error_msg: Optional[str] = None


async def send_streaming_request(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> RequestResult:
    async with semaphore:
        result = RequestResult()
        start_time = time.perf_counter()
        first_token_time = None
        last_token_time = None

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": PROMPT_TEXT}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
                stream_options={"include_usage": True},
            )

            tokens_received = 0
            async for chunk in response:
                now = time.perf_counter()

                if chunk.usage:
                    result.prompt_tokens = chunk.usage.prompt_tokens
                    result.completion_tokens = chunk.usage.completion_tokens

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    # Extract text from either normal content or reasoning/thinking stream
                    token_text = (
                        getattr(delta, "content", None)
                        or getattr(delta, "reasoning_content", None)
                        or getattr(delta, "reason", None)
                    )

                    if token_text:
                        tokens_received += 1
                        if first_token_time is None:
                            first_token_time = now
                            result.ttft_sec = first_token_time - start_time
                        else:
                            result.itl_list.append(now - last_token_time)
                        last_token_time = now

            end_time = time.perf_counter()
            result.latency_sec = end_time - start_time

            if result.completion_tokens == 0:
                result.completion_tokens = tokens_received

        except Exception as e:
            result.success = False
            result.error_msg = str(e)
        finally:
            pbar.update(1)

        return result

async def main():
    print(f" Connecting to: {BASE_URL}")
    print(f" Target Model: {MODEL_NAME}")
    print(f" Concurrency: {CONCURRENCY} | Total Requests: {NUM_PROMPTS} | Max Tokens: {MAX_TOKENS}")
    print("=" * 65)

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    # 1. Warmup step (avoids JIT / initial allocation skew)
    print(" Running warmup request...")
    try:
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )
        print(" Warmup completed.\n")
    except Exception as e:
        print(f" Warmup warning/error: {e}\n")

    # 2. Main Benchmark Loop
    semaphore = asyncio.Semaphore(CONCURRENCY)
    pbar = tqdm(total=NUM_PROMPTS, desc="Benchmarking", unit="req")
    
    wall_start = time.perf_counter()
    tasks = [
        send_streaming_request(client, semaphore, pbar)
        for _ in range(NUM_PROMPTS)
    ]
    results: List[RequestResult] = await asyncio.gather(*tasks)
    wall_time = time.perf_counter()
    wall_duration = wall_time - wall_start
    pbar.close()

    # -----------------------------------------------------------------------------
    # Aggregate Metrics Calculation
    # -----------------------------------------------------------------------------
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        print("\n All requests failed. Check endpoint and model name.")
        if failed:
            print(f"Sample error: {failed[0].error_msg}")
        return

    total_output_tokens = sum(r.completion_tokens for r in successful)
    total_input_tokens = sum(r.prompt_tokens for r in successful)
    total_tokens = total_input_tokens + total_output_tokens

    gen_throughput = total_output_tokens / wall_duration
    total_throughput = total_tokens / wall_duration
    req_throughput = len(successful) / wall_duration

    latencies = [r.latency_sec for r in successful]
    ttfts = [r.ttft_sec for r in successful if r.ttft_sec > 0]
    
    all_itls = []
    for r in successful:
        all_itls.extend(r.itl_list)

    avg_latency = statistics.mean(latencies)
    p50_latency = statistics.median(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)

    avg_ttft_ms = (statistics.mean(ttfts) * 1000) if ttfts else 0.0
    p50_ttft_ms = (statistics.median(ttfts) * 1000) if ttfts else 0.0
    avg_itl_ms = (statistics.mean(all_itls) * 1000) if all_itls else 0.0

    print("\n" + "=" * 65)
    print("                    BENCHMARK RESULTS                    ")
    print("=" * 65)
    print(f" Wall Clock Time:             {wall_duration:.2f} s")
    print(f" Requests Succeeded:          {len(successful)}/{NUM_PROMPTS} ({(len(successful)/NUM_PROMPTS)*100:.1f}%)")
    print(f" Total Input (Prompt) Tokens: {total_input_tokens:,}")
    print(f" Total Output Tokens:         {total_output_tokens:,}")
    print("-" * 65)
    print(f"  Output Generation TPS:     {gen_throughput:,.2f} tokens/s")
    print(f"  Total (In+Out) TPS:        {total_throughput:,.2f} tokens/s")
    print(f"  Request Throughput:        {req_throughput:.2f} req/s")
    print("-" * 65)
    print(f" Avg End-to-End Latency:      {avg_latency:.2f} s (P50: {p50_latency:.2f}s | P95: {p95_latency:.2f}s)")
    print(f" Avg Time-to-First-Token:     {avg_ttft_ms:.1f} ms (P50: {p50_ttft_ms:.1f} ms)")
    stream_tps_str = f"{1000 / avg_itl_ms:.1f}" if avg_itl_ms > 0 else "N/A"
    print(f" Avg Inter-Token Latency:     {avg_itl_ms:.1f} ms (~{stream_tps_str} tps/stream)")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())