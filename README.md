# Aura RL Training Cluster

A learning-focused project for understanding how to build RL/RLHF training infrastructure using Ray, Kubernetes (GKE), and Pulumi.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GKE Cluster                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Ray Cluster                                   │  │
│  │  ┌─────────────┐    ┌─────────────────────────────────────────────┐   │  │
│  │  │  Ray Head   │    │              Ray Workers                    │   │  │
│  │  │  ┌───────┐  │    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │  │
│  │  │  │ GCS   │  │────│  │CPU Worker│  │GPU Worker│  │GPU Worker│  │   │  │
│  │  │  │ Dash  │  │    │  │(rollouts)│  │(training)│  │(training)│  │   │  │
│  │  │  └───────┘  │    │  └──────────┘  └──────────┘  └──────────┘  │   │  │
│  │  └─────────────┘    └─────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │  CPU Node Pool             │  │  GPU Node Pool                      │   │
│  │  n2-standard-4 (preempt)   │  │  n1-standard-8 + T4 (preempt)       │   │
│  │  1-3 nodes                 │  │  0-2 nodes                          │   │
│  └─────────────────────────────┘  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
.
├── docs/
│   └── K8S_FUNDAMENTALS.md     # K8s concepts mapped to ML workloads
├── pulumi/
│   ├── Pulumi.yaml             # Project definition
│   ├── Pulumi.dev.yaml         # Dev stack config
│   ├── index.ts                # Main infrastructure code
│   ├── components/
│   │   ├── gke-cluster.ts      # GKE cluster component
│   │   ├── node-pools.ts       # CPU/GPU node pools
│   │   └── kuberay.ts          # KubeRay operator
│   └── config/
│       └── ray-cluster.yaml    # Sample Ray cluster manifest
└── ray/
    ├── requirements.txt        # Python dependencies
    └── example_rlhf_training.py # Sample RLHF training script
```

## Key Concepts

### Kubernetes Fundamentals

| Concept | Purpose | ML Use Case |
|---------|---------|-------------|
| **Pod** | Smallest deployable unit | Single training process |
| **Node** | Worker machine (VM) | CPU or GPU instance |
| **Node Pool** | Group of identical nodes | Separate CPU/GPU resources |
| **Deployment** | Manages pod replicas | Inference servers |
| **Job** | Run-to-completion pods | Training runs |
| **Service** | Stable network endpoint | Ray dashboard access |

See [docs/K8S_FUNDAMENTALS.md](docs/K8S_FUNDAMENTALS.md) for detailed explanations.

### GPU Scheduling

```yaml
# Request a GPU in your pod spec
resources:
  requests:
    nvidia.com/gpu: 1
  limits:
    nvidia.com/gpu: 1
```

Key points:
- GPUs are non-sharable (one pod = one GPU, unless using MIG/time-slicing)
- GPU nodes are tainted to prevent non-GPU workloads
- Use tolerations in your pod spec to run on GPU nodes

### Ray on Kubernetes

**KubeRay Operator** manages Ray clusters as Kubernetes custom resources:

1. Install operator: Helm chart creates controller deployment
2. Create RayCluster: Define head + worker specs
3. Operator creates: Pods, services, autoscaling

## Quick Start

### Prerequisites

- GCP project with billing enabled
- `gcloud` CLI installed and authenticated
- `pulumi` CLI installed
- `kubectl` installed
- Node.js 18+ (for Pulumi TypeScript)

### Deploy Infrastructure

```bash
# Navigate to Pulumi directory
cd pulumi

# Install dependencies
npm install

# Configure your GCP project
pulumi config set gcp:project YOUR_PROJECT_ID

# Preview changes
pulumi preview

# Deploy
pulumi up
```

### Connect to Cluster

```bash
# Get kubectl credentials
gcloud container clusters get-credentials aura-rl-dev --zone us-central1-a

# Verify nodes
kubectl get nodes

# Check KubeRay operator
kubectl get pods -n ray-system
```

### Deploy Ray Cluster

```bash
# Apply Ray cluster manifest
kubectl apply -f config/ray-cluster.yaml

# Check Ray cluster status
kubectl get rayclusters

# Port-forward to Ray dashboard
kubectl port-forward svc/rlhf-cluster-head-svc 8265:8265
# Open http://localhost:8265
```

### Submit Training Job

```bash
# Submit job to Ray cluster
ray job submit \
  --address http://localhost:8265 \
  --working-dir ./ray \
  -- python example_rlhf_training.py
```

## Cost Optimization

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| Preemptible GPUs | 60-70% | May be terminated |
| Scale to zero | 100% when idle | Cold start latency |
| T4 vs A100 | ~10x cheaper | Slower training |
| LoRA fine-tuning | Less GPU memory | Slightly lower quality |

## Team Discussion Points

1. **GPU Utilization**: What's our target? <50% = wasting money
2. **Checkpointing**: How often? Where stored? (GCS recommended)
3. **Node Pool Strategy**: One pool or specialized pools?
4. **Preemption Handling**: Retry logic, checkpoint frequency
5. **Autoscaling Triggers**: When to scale up/down?

## Resources

- [Ray Documentation](https://docs.ray.io/)
- [KubeRay GitHub](https://github.com/ray-project/kuberay)
- [GKE GPU Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)
- [Pulumi GCP Provider](https://www.pulumi.com/registry/packages/gcp/)
- [TRL Library (RLHF)](https://huggingface.co/docs/trl/)
