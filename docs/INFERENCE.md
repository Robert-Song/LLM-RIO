# Inference API

All inference endpoints require `Authorization: Bearer …`. User keys must have a model
grant. TA/admin keys have their documented administrative privileges. Quotas reserve
estimated work atomically, settle reported usage once, and release unused reservations
on failure/disconnect. Authoritative zero usage remains zero.

```sh
export LLMRIO_API_URL=http://127.0.0.1:8002
# Set LLMRIO_API_KEY to your issued key; do not put it in source control.
curl -fsS "$LLMRIO_API_URL/v1/models" \
  -H "Authorization: Bearer $LLMRIO_API_KEY"
curl -fsS "$LLMRIO_API_URL/v1/chat/completions" \
  -H "Authorization: Bearer $LLMRIO_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"example","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
curl -N "$LLMRIO_API_URL/v1/chat/completions" \
  -H "Authorization: Bearer $LLMRIO_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"example","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true,"stream_options":{"include_usage":true}}'
```

Streaming uses SSE and `[DONE]`. Tool and reasoning fields are forwarded through the
router; model and engine support still determine their behavior. Cancellation releases
leases and accounting reservations. `GET /v1/me/usage` reports the caller's usage.

A model can appear in `/v1/models` with `callable=false` when its profiles need validation.
Unavailable artifacts, changed local files, invalid profiles, access restrictions and
quota exhaustion produce explicit errors. Clients should inspect HTTP status and the
structured error payload, and avoid blindly retrying a streamed request after output.
Use `X-Test-Run-ID` to associate qualification requests with authenticated
`GET /admin/requests?test_run_id=…` accounting records.
