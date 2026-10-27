# Optional AWS deployment plan

This is a future design only. The repository creates no AWS account resources, credentials, images, buckets, or cloud infrastructure. Keep local validation and cost estimates ahead of any implementation.

## Suggested stages

1. **Containerize and measure locally.** Freeze Python/Node dependencies, add CI checks, pin model revisions, test backup and restore for PostgreSQL and Qdrant, and collect a representative request/traffic profile.
2. **Publish artifacts.** Build and scan API/frontend images, publish to a private registry, and store static dashboard assets in object storage behind a CDN. Keep secrets in a managed secret store.
3. **Start with CPU services.** Run the stateless API and frontend on a small container service, PostgreSQL as managed relational storage, and Redis only if history reads benefit from measured caching. Select a managed vector option or deploy Qdrant only after confirming operational and data-migration requirements.
4. **Add on-demand GPU serving.** Put vLLM on a GPU-backed VM or container capacity only when workload warrants it. Keep model weights in a persistent cache, expose the model service on a private network, and stop/scale the GPU worker down when unused. Measure cold-start time against cost before choosing a scaling policy.
5. **Observe and recover.** Export service metrics, centralize structured logs, alert on API availability and database/storage health, and rehearse database/vector backups and recovery.

## Target topology

```text
Browser → CDN/static dashboard → HTTPS API gateway/load balancer → CPU API workers
                                                           ├─ managed PostgreSQL
                                                           ├─ Redis cache (optional)
                                                           ├─ private vector service
                                                           └─ on-demand vLLM GPU service
Container registry + secret store + logs/metrics + backup storage
```

## Cost and security controls

- Start with CPU-only capacity and a capped budget; forecast database, vector storage, object storage, network transfer, logs, and GPU runtime separately.
- Keep GPU capacity off when there is no active workload. Test whether a smaller quantized model, lower concurrency, or batch processing meets the same objective before scaling GPU hardware.
- Put the model endpoint, database, Redis, and vector store on private networking. Expose only the authenticated API and static frontend.
- Use short-lived identity and managed secrets; rotate credentials. Enable TLS, request limits, authorization, and audit logging at the edge.
- Set retention limits for prompts and evaluation data, control access to uploaded documents, and define deletion/backup policies before importing private data.
- Use infrastructure-as-code and an explicit teardown process. No AWS resource should be created until the local system, recovery plan, and monthly estimate are reviewed.

AWS service choices and pricing change; re-check current service capabilities and regional costs when this phase is authorized and scheduled.
