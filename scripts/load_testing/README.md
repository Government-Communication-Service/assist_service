# Load Testing

[Locust](https://locust.io)-based load testing suite for the Copilot API.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Access to a running Copilot API environment
- AWS credentials / VPN access if targeting dev/test

## Setup

1. Populate `.env.dev` (or `.env`) and set the required values:

   ```
   BASE_URL=https://dev.api.copilot.gcs.civilservice.gov.uk
   FRONTEND_URL=https://connect.communications.gov.uk
   AUTH_SECRET_KEY=<the AUTH_SECRET_KEY from the target environment>
   STYLE_GUIDE_USE_CASE_UUID=<UUID of the "GOV.UK style guide checker" use case on the target env>
   ```

   `FRONTEND_URL` is used to populate `Origin` / `Referer` headers so requests
   pass the AWS WAF `AWSManagedRulesCommonRuleSet` without disabling it. It can probably be any valid URL.

2. Place test data files in `data/ignored/load_testing/` (repo root):

   | File | Used by |
   |---|---|
   | `small.txt` | document upload tasks |
   | `medium.txt` | document upload tasks |
   | `large.txt` | document upload tasks |
   | `sample.csv` | document upload tasks |
   | `great-campaign.pptx` | audience builder tasks (optional) |
   | `pre-school-parents.pptx` | audience builder tasks (optional) |
   | `ravening-hordes.txt` | audience builder tasks (optional) |

   Audience builder files are optional — the suite falls back to `medium.txt` if none are present.

3. Install dependencies:

   ```bash
   cd scripts/load_testing
   uv sync
   ```

## Running

### Web UI (interactive)

```bash
cd scripts/load_testing
uv run locust -f locustfile.py --class-picker
```

Open http://localhost:8089, select a profile, set users/ramp, and start.

### Headless (CI / scripted)

```bash
cd scripts/load_testing

uv run locust -f locustfile.py CheapSoakUser      -u 10 -r 2  -t 4h  --headless --csv=results/cheap_soak
uv run locust -f locustfile.py DocumentStressUser  -u 8  -r 1  -t 2h  --headless --csv=results/doc_stress
uv run locust -f locustfile.py RagLoadUser         -u 4  -r 1  -t 1h  --headless --csv=results/rag_load
uv run locust -f locustfile.py FullMixUser         -u 12 -r 2  -t 2h  --headless --csv=results/full_mix
uv run locust -f locustfile.py AudienceBuilderUser -u 2  -r 1  -t 1h  --headless --csv=results/ab_pipeline
uv run locust -f locustfile.py StyleGuideUser      -u 2  -r 1  -t 1h  --headless --csv=results/style_guide
```

## Profiles

| Profile | Purpose | Bedrock cost | Recommended params |
|---|---|---|---|
| `CheapSoakUser` | Long soak to detect memory growth — no LLM calls | None | `-u 10 -r 2 -t 4h` |
| `DocumentStressUser` | Heavy upload focus — targets the unstructured parsing path | None | `-u 8 -r 1 -t 2h` |
| `RagLoadUser` | RAG chat — exercises Bedrock + OpenSearch under load | Medium | `-u 4 -r 1 -t 1h` |
| `FullMixUser` | Production-like traffic mix across all areas | Medium | `-u 12 -r 2 -t 2h` |
| `AudienceBuilderUser` | Full audience builder pipeline (upload → analyse → citations → save) | High | `-u 2 -r 1 -t 1h` |
| `StyleGuideUser` | Style guide checks via the chat endpoint | Medium | `-u 2 -r 1 -t 1h` |

Keep `AudienceBuilderUser` and `StyleGuideUser` at very low concurrency — each task invokes Bedrock heavily.

## AWS WAF compatibility

Requests include the full set of browser-like headers (`Origin`, `Referer`, `Accept`,
`Sec-Fetch-*`, `Sec-Ch-Ua-*`) expected by the `AWSManagedRulesCommonRuleSet`.
