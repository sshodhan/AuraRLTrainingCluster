# Aura Ecosystem — From Inference Lab to Production

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AURA ECOSYSTEM                                     │
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│  │  LEARN        │    │  TRAIN            │    │  SERVE                  │   │
│  │              │    │                  │    │                          │   │
│  │  Inference   │───▶│  RL Training     │───▶│  AuraApp Production     │   │
│  │  Benchmarks  │    │  Cluster         │    │                          │   │
│  │              │    │                  │    │  ┌────────────────────┐  │   │
│  │  - vLLM      │    │  - GKE + Ray     │    │  │ AuraStylistAgent  │  │   │
│  │  - TTFT      │    │  - RLHF          │    │  │ (Web)             │  │   │
│  │  - KV cache  │    │  - Fine-tuning   │    │  ├────────────────────┤  │   │
│  │  - Batching  │    │  - LoRA          │    │  │ aura-pipeline-svc │  │   │
│  │  - Quant     │    │                  │    │  │ (Backend)         │  │   │
│  │              │    │                  │    │  ├────────────────────┤  │   │
│  │              │    │                  │    │  │ aura-android       │  │   │
│  │              │    │                  │    │  │ (Mobile)          │  │   │
│  └──────────────┘    └──────────────────┘    │  └────────────────────┘  │   │
│                                              └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Repositories

| Repo | Purpose | Status |
|------|---------|--------|
| [AuraInferenceBenchmarks](https://github.com/sshodhan/AuraInferenceBenchmarks) | Learn inference concepts hands-on (vLLM, TTFT, KV cache, batching, quantization) | New — inference lab goes here |
| [AuraRLTrainingCluster](https://github.com/sshodhan/AuraRLTrainingCluster) | GKE + Ray infrastructure for distributed RL/RLHF training | Infrastructure defined |
| [AuraStylistAgent](https://github.com/sshodhan/AuraStylistAgent) | React web app — AI styling assistant (currently uses Gemini API) | Live on Vercel |
| [aura-pipeline-service](https://github.com/sshodhan/aura-pipeline-service) | Backend pipeline — pre-computes 40-60K outfit bundles nightly, cached in Redis | Running on Cloud Run |
| [aura-stylist-android](https://github.com/sshodhan/aura-stylist-android) | Android client for Aura Stylist | In development |

## How AuraApp Works Today

```
User Request ──▶ aura-pipeline-service ──▶ Gemini API ──▶ Outfit Recommendations
                         │                                         │
                         ▼                                         ▼
                   Redis Cache ◀──── Pre-computed nightly ◀──── Gemini generates
                   (sub-20ms)        for 12 US cities           outfit bundles
```

**Current architecture:**
- **Gemini API** generates all outfit recommendations
- **aura-pipeline-service** pre-computes ~40-60K outfits nightly at 2 AM UTC
- Results cached in **Redis** (Google Cloud Memorystore) for sub-20ms retrieval
- V2 API adds seasonal context, confidence scores, hero images, catalog hints

**Current costs/constraints:**
- Gemini API costs per generation call
- Nightly batch job processes all 12 cities × personas × occasions × vibes
- No control over model behavior beyond prompt engineering
- Dependent on Google's model availability and pricing

## Why the Inference Lab Matters for AuraApp

The inference lab teaches you the exact concepts needed to eventually **self-host a model for AuraApp**:

### Lab → AuraApp Connection

| Inference Lab Concept | How It Applies to AuraApp |
|---|---|
| **Lab 1: Deploy with vLLM** | Self-host a fine-tuned fashion model instead of calling Gemini API |
| **Lab 2: KV Cache & TTFT** | Understand prefill cost when generating outfit descriptions (long system prompts with style rules) |
| **Lab 3: Batching & Throughput** | The nightly pipeline generates 40-60K outfits — batching efficiency is critical for cost |
| **Lab 4: Model Size Comparison** | Choose the right model size: small/fast for real-time requests vs larger for nightly batch |
| **Lab 5: Quantization** | Run a quantized model to cut GPU costs for the nightly pipeline by ~50% |
| **Lab 6: Prompt Caching** | Style rules + city context are shared across many requests — prefix caching slashes nightly compute |

### The Key Insight

AuraApp's nightly pipeline is a **batch inference workload** — exactly what you'll benchmark in the inference lab. The concepts map directly:

```
Pipeline Today (Gemini API):
  12 cities × ~3-5K combos/city = ~40-60K Gemini API calls/night
  Cost: API pricing per call
  Control: Prompt engineering only

Pipeline Future (Self-Hosted):
  Same 40-60K generations, but on YOUR model
  vLLM server with batching (Lab 3) → higher throughput
  Prefix caching (Lab 6) → shared style rules across all requests
  Quantization (Lab 5) → fit on cheaper GPU
  Cost: GPU hours only (potentially 10-100x cheaper at scale)
  Control: Fine-tune for fashion-specific output quality
```

## Roadmap: From Learning to Production

### Phase 1 — Learn (AuraInferenceBenchmarks)
> "I understand how inference works at the hardware level."

Complete the 6 inference labs:
- [ ] Lab 1: Deploy model with vLLM
- [ ] Lab 2: Measure KV cache impact
- [ ] Lab 3: Batching & throughput
- [ ] Lab 4: Compare model sizes
- [ ] Lab 5: Quantization impact
- [ ] Lab 6: Prompt caching simulation

**Outcome:** You can explain TTFT, KV cache, batching tradeoffs, and quantization from hands-on experience.

### Phase 2 — Train (AuraRLTrainingCluster)
> "I fine-tuned a model specifically for fashion recommendations."

Use the training cluster to fine-tune a model for AuraApp:
- [ ] Collect training data from existing Gemini outputs (outfit descriptions, style reasoning)
- [ ] Fine-tune a small model (e.g., Qwen2.5-1.5B or Llama-3-8B) on fashion/styling tasks
- [ ] Use LoRA for efficient fine-tuning on limited GPU budget
- [ ] Evaluate: does the fine-tuned model match Gemini quality for outfit generation?

**Outcome:** A fashion-specialized model that generates outfit recommendations without calling an external API.

### Phase 3 — Serve (AuraApp Integration)
> "I replaced the Gemini API dependency with a self-hosted, fine-tuned model."

Deploy the fine-tuned model for AuraApp production:
- [ ] Deploy model on GKE using vLLM (or on Cloud Run with GPU)
- [ ] Apply inference lab learnings:
  - Quantize to INT8 for cost efficiency
  - Enable prefix caching (shared style rules across all cities)
  - Tune batch size for the nightly pipeline throughput
- [ ] Update `aura-pipeline-service` to call self-hosted model instead of Gemini
- [ ] A/B test: self-hosted model vs Gemini for outfit quality
- [ ] Monitor: TTFT, throughput, GPU utilization, cost per outfit

**Outcome:** AuraApp runs on your own model — cheaper, faster, and fully under your control.

### Phase 4 — Optimize with RL (Full Circle)
> "I use RLHF to improve outfit quality based on user feedback."

Close the loop with reinforcement learning:
- [ ] Collect user preference signals from the Android app (thumbs up/down, outfit selections)
- [ ] Train a reward model on user preferences
- [ ] Use RLHF (on the training cluster) to align the fashion model with real user taste
- [ ] Redeploy improved model → measure user engagement lift

**Outcome:** A continuously improving fashion model trained on real user feedback.

## Architecture Evolution

### Today: API-Dependent
```
Android App ──▶ Pipeline Service ──▶ Gemini API
                      │
                      ▼
                 Redis Cache
```

### Future: Self-Hosted + RL Loop
```
Android App ──────────────────────────────────────────────┐
      │                                                    │
      ▼                                                    ▼
Pipeline Service ──▶ Self-Hosted Model (vLLM on GKE)   User Feedback
      │                    │                               │
      ▼                    │                               ▼
 Redis Cache               │                         Reward Model
                           │                               │
                           ◀── RLHF Training (Ray) ◀───────┘
```

## Cost Comparison (Projected)

| Approach | Nightly Cost (40-60K generations) | Per-Request Latency | Model Control |
|---|---|---|---|
| **Gemini API** (today) | ~$20-50/night (depending on model) | 500-2000ms | Prompt engineering only |
| **Self-hosted FP16** (A10G) | ~$3-6/night (1 GPU × 4-8 hrs) | 50-200ms | Full fine-tuning |
| **Self-hosted INT8** (T4) | ~$1-3/night (1 GPU × 4-8 hrs) | 80-300ms | Full fine-tuning |

*Estimates depend on model size, prompt length, and output length. The inference lab will help you measure these precisely.*

## Getting Started

1. **Start with the Inference Lab** → [AuraInferenceBenchmarks](https://github.com/sshodhan/AuraInferenceBenchmarks)
   - Use Google Colab (free T4) or RunPod ($2-5/hr)
   - Complete Labs 1-6 over 2 days

2. **Then explore the Training Cluster** → [AuraRLTrainingCluster](https://github.com/sshodhan/AuraRLTrainingCluster)
   - Review `docs/K8S_FUNDAMENTALS.md` for K8s concepts
   - Study the Pulumi infrastructure code
   - Understand how Ray distributes training

3. **When ready, integrate** → Update `aura-pipeline-service` to call your own model
