"""Enterprise operational knowledge base and Golden Evaluation Dataset."""

ENTERPRISE_KNOWLEDGE_BASE = [
    {
        "doc_id": "kb_incident_response_p1",
        "text": (
            "Enterprise P1 Incident Response SLA Protocol: High-severity production outages require an incident commander "
            "assignment within 15 minutes. The reliability engineer must initiate a war room on Slack #incidents-prod. "
            "Database failover to secondary read-replicas must be executed if primary node replication lag exceeds 500ms for over 3 minutes. "
            "Root-cause analysis (RCA) document must be published within 48 business hours post-incident resolution."
        ),
    },
    {
        "doc_id": "kb_api_rate_limiting",
        "text": (
            "API Gateway Rate Limiting & Token Bucket Policy: Public tier endpoints enforce a strict token-bucket algorithm "
            "with a burst capacity of 100 requests and a sustained refill rate of 20 requests per second per IP address. "
            "Enterprise tier API keys are provisioned with 500 requests per second and isolated Redis clusters for token state tracking. "
            "Exceeding quota triggers HTTP 429 Too Many Requests with a Retry-After header specified in seconds."
        ),
    },
    {
        "doc_id": "kb_data_encryption_standard",
        "text": (
            "Data Security & Cryptographic Storage Standard: All data at rest across relational databases and object storage "
            "must be encrypted using AES-256 with AWS KMS or HashiCorp Vault managed keys. Encryption keys rotate automatically every 90 days. "
            "Data in transit requires TLS 1.3 with forward secrecy enabled. Legacy TLS 1.0 and 1.1 handshakes are immediately rejected at the Cloudflare edge."
        ),
    },
    {
        "doc_id": "kb_ml_model_deployment",
        "text": (
            "Production ML Model Deployment & Canary Rollout: All candidate model weights must pass shadow-traffic evaluation for 72 hours. "
            "Canary deployments begin with 5 percent traffic allocation. Automated rollbacks are triggered if model inference p99 latency "
            "exceeds 250ms or if prediction null rate spikes above 0.05 percent over a continuous 10-minute observation window."
        ),
    },
]

GOLDEN_BENCHMARK_QUERIES = [
    {
        "query": "What is the replication lag threshold for database failover during a P1 outage?",
        "expected_doc_id": "kb_incident_response_p1",
    },
    {
        "query": "What is the sustained refill rate and burst capacity for the public tier API token bucket?",
        "expected_doc_id": "kb_api_rate_limiting",
    },
    {
        "query": "How often do AES-256 KMS encryption keys rotate at rest?",
        "expected_doc_id": "kb_data_encryption_standard",
    },
    {
        "query": "What p99 inference latency threshold triggers an automated canary deployment rollback?",
        "expected_doc_id": "kb_ml_model_deployment",
    },
]
