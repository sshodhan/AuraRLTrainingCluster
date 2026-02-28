# Kubernetes Fundamentals for ML/RL Workloads

This document maps K8s primitives to ML training concepts. Use this to have informed conversations with your team.

---

## The Mental Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLUSTER (GKE)                                                              │
│  └── The entire K8s installation. In GKE, Google manages the control plane. │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  NODE POOL: cpu-workers                                             │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │    │
│  │  │   NODE      │  │   NODE      │  │   NODE      │  (VMs)          │    │
│  │  │ n2-standard │  │ n2-standard │  │ n2-standard │                 │    │
│  │  │   -8        │  │   -8        │  │   -8        │                 │    │
│  │  │  ┌──────┐   │  │  ┌──────┐   │  │  ┌──────┐   │                 │    │
│  │  │  │ POD  │   │  │  │ POD  │   │  │  │ POD  │   │  (Containers)   │    │
│  │  │  └──────┘   │  │  └──────┘   │  │  └──────┘   │                 │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  NODE POOL: gpu-workers                                             │    │
│  │  ┌─────────────────────┐  ┌─────────────────────┐                  │    │
│  │  │   NODE              │  │   NODE              │                  │    │
│  │  │ n1-standard-8       │  │ n1-standard-8       │                  │    │
│  │  │ + NVIDIA T4 GPU     │  │ + NVIDIA T4 GPU     │                  │    │
│  │  │  ┌──────────────┐   │  │  ┌──────────────┐   │                  │    │
│  │  │  │ POD (GPU)    │   │  │  │ POD (GPU)    │   │                  │    │
│  │  │  │ Training Job │   │  │  │ Training Job │   │                  │    │
│  │  │  └──────────────┘   │  │  └──────────────┘   │                  │    │
│  │  └─────────────────────┘  └─────────────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core K8s Primitives

### 1. Pod - The Atomic Unit

**What it is**: Smallest deployable unit. One or more containers that share storage/network.

**ML Analogy**: A single training process or inference server.

```yaml
# A pod requesting a GPU for training
apiVersion: v1
kind: Pod
metadata:
  name: llm-training-pod
spec:
  containers:
  - name: trainer
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    resources:
      requests:          # Minimum resources needed
        memory: "16Gi"
        cpu: "4"
        nvidia.com/gpu: 1   # <-- GPU request
      limits:            # Maximum resources allowed
        memory: "32Gi"
        cpu: "8"
        nvidia.com/gpu: 1   # <-- GPUs are NOT oversubscribed
    command: ["python", "train.py"]
```

**Key Insight for GPUs**:
- GPUs are **non-compressible** resources - a pod gets exactly what it requests
- Unlike CPU (which can be oversubscribed), GPU `requests` == `limits`
- One T4 GPU cannot be shared between pods (without MIG or time-slicing)

---

### 2. Node - The Machine

**What it is**: A worker machine (VM in GKE) that runs pods.

**ML Relevance**: Different node types for different workloads.

| Node Type | Use Case | Example Machine |
|-----------|----------|-----------------|
| CPU-optimized | Data preprocessing, environment simulation | n2-standard-8 |
| Memory-optimized | Large batch inference, embedding | n2-highmem-8 |
| GPU-attached | Training, inference | n1-standard-8 + T4 |
| TPU | Large-scale training | TPU v4 pod |

**Node Labels** - How you target specific hardware:
```yaml
# Node automatically gets labels like:
cloud.google.com/gke-accelerator: nvidia-tesla-t4
cloud.google.com/gke-accelerator-count: "1"
node.kubernetes.io/instance-type: n1-standard-8
```

---

### 3. Node Pool (GKE-specific) - Grouped Machines

**What it is**: A group of nodes with identical configuration.

**Why it matters for ML**:
- Separate pools for CPU vs GPU workloads
- Different scaling policies per pool
- Cost optimization (preemptible GPUs are 60-70% cheaper)

```
Cluster
├── Node Pool: "default-pool" (system workloads)
│   └── n2-standard-4 x 3
├── Node Pool: "cpu-workers" (rollout/simulation)
│   └── n2-standard-8 x 5, preemptible
└── Node Pool: "gpu-workers" (training)
    └── n1-standard-8 + T4 x 2, preemptible
```

---

### 4. Deployment vs Job vs StatefulSet

These control HOW pods are managed:

| Controller | Use Case | ML Example |
|------------|----------|------------|
| **Deployment** | Long-running, stateless, replaceable | Inference server, Ray head |
| **Job** | Run-to-completion | Training script, batch inference |
| **StatefulSet** | Stable identity, persistent storage | Distributed training with checkpoints |
| **DaemonSet** | One per node | GPU driver installer, monitoring agent |

**For RLHF specifically**:
- **Deployment**: Reward model server (always running)
- **Job**: PPO training runs (run to completion)
- **StatefulSet**: If you need persistent checkpoints with stable naming

---

### 5. Service - Network Abstraction

**What it is**: Stable endpoint to reach pods (pods come and go, services persist).

**ML Use Cases**:

```yaml
# Expose Ray Dashboard
apiVersion: v1
kind: Service
metadata:
  name: ray-dashboard
spec:
  selector:
    app: ray-head
  ports:
  - port: 8265
    targetPort: 8265
  type: LoadBalancer  # External access
---
# Internal service for reward model
apiVersion: v1
kind: Service
metadata:
  name: reward-model
spec:
  selector:
    app: reward-model
  ports:
  - port: 8000
  type: ClusterIP  # Internal only
```

---

### 6. ConfigMap & Secret - Configuration

**What they are**: Inject config and secrets into pods without baking into images.

```yaml
# Training hyperparameters
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config
data:
  config.yaml: |
    learning_rate: 0.0001
    batch_size: 32
    ppo_epochs: 4
---
# Mount in pod
spec:
  containers:
  - name: trainer
    volumeMounts:
    - name: config
      mountPath: /app/config
  volumes:
  - name: config
    configMap:
      name: training-config
```

---

### 7. PersistentVolume (PV) & PersistentVolumeClaim (PVC)

**What they are**: Storage that survives pod restarts.

**Critical for ML**:
- Model checkpoints
- Training datasets
- Logs and metrics

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-checkpoints
spec:
  accessModes:
    - ReadWriteOnce      # Single node can write
  storageClassName: standard
  resources:
    requests:
      storage: 100Gi
---
# Use in pod
spec:
  containers:
  - name: trainer
    volumeMounts:
    - name: checkpoints
      mountPath: /checkpoints
  volumes:
  - name: checkpoints
    persistentVolumeClaim:
      claimName: model-checkpoints
```

**Access Modes for Distributed Training**:
| Mode | Description | Use Case |
|------|-------------|----------|
| ReadWriteOnce (RWO) | Single node read/write | Single trainer checkpoints |
| ReadOnlyMany (ROX) | Many nodes read | Shared dataset |
| ReadWriteMany (RWX) | Many nodes read/write | Distributed checkpoints (needs NFS/Filestore) |

---

## GPU Scheduling Deep Dive

### How K8s Schedules GPUs

1. **Device Plugin**: NVIDIA device plugin runs as DaemonSet, exposes GPUs to K8s
2. **Resource Accounting**: Each GPU is `nvidia.com/gpu: 1`
3. **Scheduling**: K8s scheduler matches pod GPU requests to available nodes

```
Pod requests nvidia.com/gpu: 2
        │
        ▼
┌─────────────────────────────────────┐
│  K8s Scheduler                      │
│  - Find nodes with >= 2 free GPUs   │
│  - Check other constraints (memory) │
│  - Select best fit                  │
└─────────────────────────────────────┘
        │
        ▼
Node with 4x T4 GPUs (2 in use, 2 free) ✓
```

### GPU Constraints You'll Hit

| Constraint | What Happens | Solution |
|------------|--------------|----------|
| **No GPU available** | Pod stays Pending | Add nodes or wait |
| **GPU memory OOM** | Pod crashes | Reduce batch size, use gradient checkpointing |
| **Wrong GPU type** | Pod Pending | Use nodeSelector or nodeAffinity |
| **Preemption** | Spot/preemptible node dies | Checkpointing, retry logic |

### Targeting Specific GPU Types

```yaml
spec:
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-tesla-t4
  # OR more flexible:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: cloud.google.com/gke-accelerator
            operator: In
            values:
            - nvidia-tesla-t4
            - nvidia-tesla-a100  # Either T4 or A100
```

### GPU Memory Planning (Critical for LLMs)

| GPU | VRAM | Can Fit (fp16) |
|-----|------|----------------|
| T4 | 16GB | 7B model inference, small training |
| A10G | 24GB | 7B training, 13B inference |
| A100 40GB | 40GB | 13B training, 30B inference |
| A100 80GB | 80GB | 30B training, 70B inference |
| H100 | 80GB | Larger with better perf |

**For Llama 2 7B RLHF** (your use case):
- Inference: ~14GB (fits on T4 barely, A10G comfortable)
- Training with optimizer states: ~28GB (needs A10G or A100)
- Full RLHF (policy + ref + reward): ~42GB (A100 40GB or 2x A10G)

---

## Resource Requests vs Limits

```yaml
resources:
  requests:    # Scheduler uses this to find a node
    memory: "16Gi"
    cpu: "4"
  limits:      # Container gets killed if exceeds this
    memory: "32Gi"
    cpu: "8"
```

**Best Practices for ML**:

| Resource | Recommendation | Why |
|----------|----------------|-----|
| CPU | requests < limits | Burst for data loading |
| Memory | requests ≈ limits | OOM kills are bad for training |
| GPU | requests == limits | Can't oversubscribe anyway |

---

## Namespaces - Logical Isolation

```
cluster/
├── namespace: kube-system      (K8s internals)
├── namespace: kuberay-system   (Ray operator)
├── namespace: training-dev     (Your dev workloads)
└── namespace: training-prod    (Production training)
```

**Use for**:
- Team isolation
- Resource quotas per environment
- RBAC boundaries

```yaml
# Limit GPU usage per namespace
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: training-dev
spec:
  hard:
    requests.nvidia.com/gpu: "4"  # Max 4 GPUs in this namespace
```

---

## Taints & Tolerations - Dedicated Hardware

**Problem**: You don't want random pods on your expensive GPU nodes.

**Solution**: Taint GPU nodes, only GPU workloads tolerate the taint.

```yaml
# Node taint (applied via node pool config)
taints:
- key: nvidia.com/gpu
  value: "true"
  effect: NoSchedule

---
# Pod toleration (allows scheduling on tainted nodes)
spec:
  tolerations:
  - key: nvidia.com/gpu
    operator: Equal
    value: "true"
    effect: NoSchedule
```

---

## Practical Patterns for RLHF

### Pattern 1: Separate Pools for Actor/Learner

```
┌─────────────────────────────────────────────────────────┐
│  Actor Pool (CPU) - Environment Rollouts                │
│  - Many cheap CPU nodes                                 │
│  - Preemptible OK (stateless)                          │
│  - Generate experiences, send to learner                │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼ Experiences (states, actions, rewards)
┌─────────────────────────────────────────────────────────┐
│  Learner Pool (GPU) - Policy Updates                    │
│  - Fewer expensive GPU nodes                            │
│  - Preemptible OK if checkpointing                     │
│  - PPO updates, model training                          │
└─────────────────────────────────────────────────────────┘
```

### Pattern 2: Co-located Inference + Training

For RLHF where you need policy inference + reward model + training:

```yaml
# Single GPU pod running all components
spec:
  containers:
  - name: rlhf-trainer
    resources:
      requests:
        nvidia.com/gpu: 1
    env:
    - name: CUDA_VISIBLE_DEVICES
      value: "0"
    # Use torch.cuda.memory_fraction to split GPU memory
```

---

## Questions to Ask Your Team

1. **"What's our GPU utilization target?"** - If it's <50%, you're wasting money
2. **"How do we handle preemption?"** - Checkpointing strategy?
3. **"What's our node pool strategy?"** - One big pool or specialized pools?
4. **"How do we manage GPU driver versions?"** - GKE auto-updates or pinned?
5. **"What's our storage strategy for checkpoints?"** - GCS, Filestore, or local SSD?

---

## Next Steps

1. **Hands-on**: We'll create Pulumi code for a GKE cluster with GPU pools
2. **Deploy**: Install KubeRay operator and create a Ray cluster
3. **Train**: Run a sample RLHF-style workload

Ready to build the infrastructure?
