# Security notes

- Bind the API to loopback by default. Use the lab's authenticated TLS reverse proxy for remote
  access; do not expose a plaintext management API to an untrusted network.
- SQLite stores an Argon2id verifier plus an encrypted recoverable copy of every API key.
  Authenticated administrators can list complete keys, as explicitly required for this lab.
- The encryption key is generated automatically beside the database as
  `state/.llm-rio-api-key-vault`. Keep the database and vault together for backup, restrict both
  to the service account, and never commit either one.
- When the database has no keys, startup creates the initial administrator before accepting HTTP
  requests and prints its complete key. Administrators can create additional independent admin
  keys at any time; LLM-RIO protects the final active administrator from deletion or revocation.
- Local `llmctl` management recovers an active admin credential only when targeting loopback and
  only from the database/vault protected by service-account filesystem permissions. Remote
  management always requires an explicitly supplied `LLMRIO_API_KEY`; management HTTP routes
  never become unauthenticated.
- Set `.env` permissions to the service account and never commit Hugging Face access tokens.
- Workers bind private loopback ports and use a per-process-lifetime internal bearer token.
- Registration accepts Hugging Face repository references, not arbitrary launch commands. Engine
  arguments come only from profiles produced by the local validator.
- LLM-RIO deliberately does not repair drivers, load kernel modules, elevate privileges, run
  containers, or coordinate other hosts.

