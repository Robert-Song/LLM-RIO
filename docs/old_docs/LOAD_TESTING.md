# Endpoint load testing

Run these commands only on the target HPC or another machine that can reach the endpoint. The
`model` field is the LLM-RIO catalog nickname, not the Hugging Face repository name. The examples
below assume Qwen was registered with the nickname `qwen3.6-27b-nvfp4`.

Set the endpoint and disposable test key without placing the key in a command-line argument:

```bash
export LLMRIO_API_URL='http://aiforge.cs.purdue.edu:8002'
export LLMRIO_API_KEY='rio_43e22ad6dc6d_0ZKev_Uydo3GsuC2hzhzjA3UQk54FsdvTBA7D-cfois'
export LLMRIO_MODEL='qwen3.6-27b-nvfp4'
```

Confirm the nickname is available to the key:

```bash
curl --fail-with-body --silent --show-error \
  "$LLMRIO_API_URL/v1/models" \
  -H "Authorization: Bearer $LLMRIO_API_KEY"
```

## One Qwen request

```bash
curl --fail-with-body --silent --show-error \
  "$LLMRIO_API_URL/v1/chat/completions" \
  -H "Authorization: Bearer $LLMRIO_API_KEY" \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "qwen3.6-27b-nvfp4",
    "messages": [
      {"role": "user", "content": "Explain continuous batching in three concise paragraphs."}
    ],
    "temperature": 0.2,
    "max_tokens": 256
  }'
```

If it has not been registered yet, a TA or administrator can start registration with:

```bash
export LLMRIO_API_URL='http://aiforge.cs.purdue.edu:8002'
export LLMRIO_API_KEY='replace-with-test-key'
./llmctl models add qwen3.6-27b-nvfp4 nvidia/Qwen3.6-27B-NVFP4
```

TA and administrator keys can use every available model without grants. To give a student key
access to an already registered model, use the API-key nickname (or its complete key) and the model
nickname:

```bash
./llmctl keys list
./llmctl models grant student-a qwen3.6-27b-nvfp4
./llmctl models access student-a
```

The grant command adds access without removing the student's existing models. Use `models revoke`
to remove specific nicknames. Internal database UUIDs are not required. For a newly registered
model, wait for its returned job to finish before inference.

## Concurrent tests

The load tester requires only the project's existing `httpx` dependency. It never writes generated
text to its metrics file. Its optional output contains per-request timing, token counts, HTTP
status, and truncated errors only.

Start with a small check:

```bash
uv run python scripts/load_test.py \
  --concurrency 8 \
  --requests-per-worker 2 \
  --prompt-tokens 64 \
  --max-tokens 128
```

Then ramp Qwen to 128 concurrent in-flight requests. Each stage completes before the next begins;
the 128 stage sends 512 requests in four bounded waves:

```bash
uv run python scripts/load_test.py \
  --ramp 1,8,16,32,64,128 \
  --requests-per-worker 4 \
  --prompt-tokens 256 \
  --max-tokens 512 \
  --timeout 900 \
  --output diagnostics/qwen36-ramp.json
```

The ramp stops before a higher stage when more than 5% of a stage fails. To test KV-cache pressure
separately, keep concurrency at 128 and increase the approximate prompt length:

```bash
uv run python scripts/load_test.py \
  --concurrency 128 \
  --requests-per-worker 2 \
  --prompt-tokens 4096 \
  --max-tokens 256 \
  --timeout 1200 \
  --output diagnostics/qwen36-kv-pressure.json
```

The prompt length is approximate because the test intentionally does not download or initialize a
tokenizer. The response usage values reported by the server are the authoritative token counts.
