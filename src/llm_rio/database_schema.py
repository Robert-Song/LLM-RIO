"""SQLite schema; changes must remain compatible with existing host databases."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS quota_accounts (
    id TEXT PRIMARY KEY,
    nickname TEXT NOT NULL UNIQUE,
    balance_tokens INTEGER NOT NULL CHECK (balance_tokens >= 0),
    limit_tokens INTEGER NOT NULL CHECK (limit_tokens >= 0),
    usage_baseline_tokens INTEGER NOT NULL DEFAULT 0,
    usage_reset_at TEXT,
    unlimited INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    nickname TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL CHECK (role IN ('user', 'ta', 'admin')),
    quota_account_id TEXT NOT NULL REFERENCES quota_accounts(id),
    token_prefix TEXT NOT NULL UNIQUE,
    token_hash TEXT NOT NULL,
    encrypted_api_key TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_used_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(token_prefix) WHERE active = 1;

CREATE TABLE IF NOT EXISTS model_catalog (
    id TEXT PRIMARY KEY,
    nickname TEXT NOT NULL UNIQUE,
    huggingface_repo TEXT NOT NULL,
    requested_revision TEXT,
    resolved_revision TEXT,
    state TEXT NOT NULL,
    artifact_path TEXT,
    artifact_hashes_json TEXT NOT NULL DEFAULT '[]',
    capabilities_json TEXT NOT NULL DEFAULT '[]',
    request_limits_json TEXT NOT NULL DEFAULT '{}',
    request_defaults_json TEXT NOT NULL DEFAULT '{}',
    source_model_id TEXT REFERENCES model_catalog(id),
    created_by_key_id TEXT NOT NULL REFERENCES api_keys(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_grants (
    key_id TEXT NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL REFERENCES model_catalog(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (key_id, model_id)
);

CREATE TABLE IF NOT EXISTS model_jobs (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    state TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress_json TEXT NOT NULL DEFAULT '{}',
    failure_json TEXT,
    requested_grants_json TEXT NOT NULL DEFAULT '[]',
    validation_overrides_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_verification_jobs (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    backend TEXT NOT NULL CHECK (backend IN ('kvcached')),
    state TEXT NOT NULL CHECK (state IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED')),
    stage TEXT NOT NULL,
    progress_json TEXT NOT NULL DEFAULT '{}',
    failure_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_verification_jobs_model
ON model_verification_jobs(model_id, created_at DESC);


CREATE TABLE IF NOT EXISTS model_profiles (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    machine_fingerprint TEXT NOT NULL,
    profile_key TEXT NOT NULL UNIQUE,
    profile_json TEXT NOT NULL,
    verified_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_profiles_model_machine
ON model_profiles(model_id, machine_fingerprint) WHERE active = 1;

CREATE TABLE IF NOT EXISTS quota_reservations (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    idempotency_hash TEXT NOT NULL,
    account_id TEXT NOT NULL REFERENCES quota_accounts(id),
    key_id TEXT NOT NULL REFERENCES api_keys(id),
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    reserved_tokens INTEGER NOT NULL CHECK (reserved_tokens >= 0),
    actual_tokens INTEGER,
    state TEXT NOT NULL CHECK (state IN ('RESERVED', 'SETTLED', 'RELEASED')),
    created_at TEXT NOT NULL,
    settled_at TEXT,
    UNIQUE (key_id, idempotency_hash)
);

CREATE TABLE IF NOT EXISTS quota_ledger (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES quota_accounts(id),
    reservation_id TEXT REFERENCES quota_reservations(id),
    delta_tokens INTEGER NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (reservation_id, reason)
);

CREATE TABLE IF NOT EXISTS inference_requests (
    id TEXT PRIMARY KEY,
    key_id TEXT NOT NULL REFERENCES api_keys(id),
    account_id TEXT NOT NULL REFERENCES quota_accounts(id),
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    reservation_id TEXT NOT NULL REFERENCES quota_reservations(id),
    worker_id TEXT,
    state TEXT NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    estimated_prompt_tokens INTEGER,
    actual_prompt_tokens INTEGER,
    actual_completion_tokens INTEGER,
    error_code TEXT,
    test_run_id TEXT,
    client_worker TEXT,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    completion_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    admitted_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS usage_summary_periods (
    window TEXT PRIMARY KEY CHECK (window IN ('current', 'total')),
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    summarized_at TEXT NOT NULL,
    request_count INTEGER NOT NULL CHECK (request_count >= 0),
    successful_requests INTEGER NOT NULL CHECK (successful_requests >= 0),
    failed_requests INTEGER NOT NULL CHECK (failed_requests >= 0),
    reserved_tokens INTEGER NOT NULL CHECK (reserved_tokens >= 0),
    charged_tokens INTEGER NOT NULL CHECK (charged_tokens >= 0),
    prompt_tokens INTEGER NOT NULL CHECK (prompt_tokens >= 0),
    completion_tokens INTEGER NOT NULL CHECK (completion_tokens >= 0),
    output_tokens_for_rate INTEGER NOT NULL CHECK (output_tokens_for_rate >= 0),
    active_output_seconds REAL NOT NULL CHECK (active_output_seconds >= 0),
    timed_requests INTEGER NOT NULL CHECK (timed_requests >= 0)
);

CREATE TABLE IF NOT EXISTS usage_summaries (
    window TEXT NOT NULL CHECK (window IN ('current', 'total')),
    account_id TEXT NOT NULL REFERENCES quota_accounts(id),
    key_id TEXT NOT NULL REFERENCES api_keys(id),
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    request_count INTEGER NOT NULL CHECK (request_count >= 0),
    successful_requests INTEGER NOT NULL CHECK (successful_requests >= 0),
    failed_requests INTEGER NOT NULL CHECK (failed_requests >= 0),
    reserved_tokens INTEGER NOT NULL CHECK (reserved_tokens >= 0),
    charged_tokens INTEGER NOT NULL CHECK (charged_tokens >= 0),
    prompt_tokens INTEGER NOT NULL CHECK (prompt_tokens >= 0),
    completion_tokens INTEGER NOT NULL CHECK (completion_tokens >= 0),
    output_tokens_for_rate INTEGER NOT NULL CHECK (output_tokens_for_rate >= 0),
    active_output_seconds REAL NOT NULL CHECK (active_output_seconds >= 0),
    timed_requests INTEGER NOT NULL CHECK (timed_requests >= 0),
    first_completed_at TEXT,
    last_completed_at TEXT,
    PRIMARY KEY (window, account_id, key_id, model_id)
);
CREATE INDEX IF NOT EXISTS idx_usage_summaries_window_model
ON usage_summaries(window, model_id);
CREATE VIEW IF NOT EXISTS account_lifetime_usage AS
SELECT account_id, SUM(charged_tokens) AS charged_tokens,
       SUM(request_count) AS settled_requests
  FROM (
        SELECT account_id, COALESCE(actual_tokens, 0) AS charged_tokens,
               1 AS request_count
          FROM quota_reservations WHERE state = 'SETTLED'
        UNION ALL
        SELECT account_id, charged_tokens, request_count
          FROM usage_summaries WHERE window = 'total'
       )
 GROUP BY account_id;
CREATE VIEW IF NOT EXISTS key_lifetime_usage AS
SELECT key_id, SUM(charged_tokens) AS charged_tokens,
       SUM(request_count) AS settled_requests
  FROM (
        SELECT key_id, COALESCE(actual_tokens, 0) AS charged_tokens,
               1 AS request_count
          FROM quota_reservations WHERE state = 'SETTLED'
        UNION ALL
        SELECT key_id, charged_tokens, request_count
          FROM usage_summaries WHERE window = 'total'
       )
 GROUP BY key_id;

CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    profile_id TEXT NOT NULL REFERENCES model_profiles(id),
    gpu_uuids_json TEXT NOT NULL,
    port INTEGER NOT NULL,
    pid INTEGER,
    state TEXT NOT NULL,
    host_cache_accounted_mib REAL NOT NULL DEFAULT 0,
    host_cache_accounting_source TEXT,
    process_rss_mib REAL NOT NULL DEFAULT 0,
    process_pss_mib REAL NOT NULL DEFAULT 0,
    process_swap_mib REAL NOT NULL DEFAULT 0,
    last_cache_eviction_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_id TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS service_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    mode TEXT NOT NULL,
    machine_fingerprint TEXT,
    updated_at TEXT NOT NULL
);
INSERT OR IGNORE INTO service_state(singleton, mode, updated_at)
VALUES (1, 'ACTIVE', CURRENT_TIMESTAMP);
"""
