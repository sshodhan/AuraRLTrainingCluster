import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";

// =============================================================================
// NODE POOLS COMPONENT
// =============================================================================
// This component creates specialized node pools for ML workloads.
//
// KEY CONCEPT - Node Pool Design for ML:
// 1. CPU Pool: For Ray head, data preprocessing, env simulation
// 2. GPU Pool: For model training, inference
// 3. Each pool can have different scaling, preemptibility, machine types

export interface NodePoolSpec {
    name: string;
    minNodes: number;
    maxNodes: number;
    machineType: string;
    preemptible: boolean;
    diskSizeGb?: number;
    diskType?: string;
    labels?: { [key: string]: string };
    taints?: Array<{
        key: string;
        value: string;
        effect: string;
    }>;
    acceleratorType?: string;
    acceleratorCount?: number;
}

export interface NodePoolsArgs {
    clusterName: pulumi.Input<string>;
    location: string;
    cpuPool: NodePoolSpec;
    gpuPool: NodePoolSpec;
}

export class NodePools extends pulumi.ComponentResource {
    public readonly cpuNodePool: gcp.container.NodePool;
    public readonly gpuNodePool: gcp.container.NodePool;

    constructor(
        name: string,
        args: NodePoolsArgs,
        opts?: pulumi.ComponentResourceOptions
    ) {
        super("custom:gke:NodePools", name, {}, opts);

        // =====================================================================
        // CPU NODE POOL
        // =====================================================================
        // For workloads that don't need GPUs:
        // - Ray head node
        // - Data loading and preprocessing
        // - Environment rollouts (for RL)
        // - Metrics and monitoring

        this.cpuNodePool = new gcp.container.NodePool(
            `${name}-cpu-pool`,
            {
                name: args.cpuPool.name,
                cluster: args.clusterName,
                location: args.location,

                // Autoscaling configuration
                // minNodes=0 allows scaling to zero when idle (cost savings)
                autoscaling: {
                    minNodeCount: args.cpuPool.minNodes,
                    maxNodeCount: args.cpuPool.maxNodes,
                },

                // Node configuration
                nodeConfig: {
                    machineType: args.cpuPool.machineType,
                    diskSizeGb: args.cpuPool.diskSizeGb ?? 100,
                    diskType: args.cpuPool.diskType ?? "pd-balanced",

                    // Preemptible VMs are 60-80% cheaper but can be terminated
                    // Good for: Batch workloads, stateless workers
                    // Bad for: Long-running jobs without checkpointing
                    preemptible: args.cpuPool.preemptible,

                    // Or use Spot VMs (newer, similar concept)
                    // spot: args.cpuPool.preemptible,

                    // OAuth scopes - what GCP APIs nodes can access
                    // "cloud-platform" gives full access (controlled by IAM)
                    oauthScopes: [
                        "https://www.googleapis.com/auth/cloud-platform",
                    ],

                    // Labels are key-value pairs attached to nodes
                    // Used for: nodeSelector in pod specs
                    labels: args.cpuPool.labels ?? {},

                    // Metadata for the underlying GCE instances
                    metadata: {
                        "disable-legacy-endpoints": "true",
                    },

                    // Shielded instance options for security
                    shieldedInstanceConfig: {
                        enableSecureBoot: true,
                        enableIntegrityMonitoring: true,
                    },
                },

                // Node management
                management: {
                    autoRepair: true,   // Replace unhealthy nodes
                    autoUpgrade: true,  // Auto-upgrade K8s version
                },

                // Upgrade settings - how nodes are upgraded
                upgradeSettings: {
                    maxSurge: 1,        // Extra nodes during upgrade
                    maxUnavailable: 0,  // Keep all nodes available
                },
            },
            { parent: this }
        );

        // =====================================================================
        // GPU NODE POOL
        // =====================================================================
        // For GPU-accelerated workloads:
        // - Model training (forward/backward pass)
        // - Inference (if GPU-bound)
        // - CUDA operations
        //
        // KEY GPU CONCEPTS:
        // - GPUs are non-sharable resources (one pod gets the whole GPU)
        // - Must use compatible machine types (n1-standard-*, a2-*, g2-*)
        // - GPU drivers are installed automatically via DaemonSet

        this.gpuNodePool = new gcp.container.NodePool(
            `${name}-gpu-pool`,
            {
                name: args.gpuPool.name,
                cluster: args.clusterName,
                location: args.location,

                // GPU nodes often have different scaling needs
                // minNodes=0 is common - spin up only when needed
                autoscaling: {
                    minNodeCount: args.gpuPool.minNodes,
                    maxNodeCount: args.gpuPool.maxNodes,
                },

                nodeConfig: {
                    machineType: args.gpuPool.machineType,
                    diskSizeGb: args.gpuPool.diskSizeGb ?? 200,  // Larger for model weights
                    diskType: args.gpuPool.diskType ?? "pd-balanced",
                    preemptible: args.gpuPool.preemptible,

                    // GPU configuration
                    // The accelerator is attached to each node in the pool
                    guestAccelerator: args.gpuPool.acceleratorType
                        ? [
                              {
                                  type: args.gpuPool.acceleratorType,
                                  count: args.gpuPool.acceleratorCount ?? 1,

                                  // GPU driver installation strategy
                                  // DEFAULT: Auto-install via GKE
                                  gpuDriverInstallationConfig: {
                                      gpuDriverVersion: "DEFAULT",
                                  },

                                  // GPU sharing config (optional, for MIG or time-slicing)
                                  // Uncomment if you want to share GPUs
                                  // gpuSharingConfig: {
                                  //     gpuSharingStrategy: "TIME_SHARING",
                                  //     maxSharedClientsPerGpu: 2,
                                  // },
                              },
                          ]
                        : [],

                    oauthScopes: [
                        "https://www.googleapis.com/auth/cloud-platform",
                    ],

                    labels: {
                        ...args.gpuPool.labels,
                        // Add GPU-specific labels for easy targeting
                        "gpu-type": args.gpuPool.acceleratorType ?? "none",
                    },

                    // Taints prevent non-GPU workloads from being scheduled here
                    // Only pods with matching tolerations can run on these nodes
                    //
                    // WHY TAINT GPU NODES?
                    // 1. GPU nodes are expensive
                    // 2. Random pods shouldn't land here and waste resources
                    // 3. Forces explicit GPU request in pod spec
                    taints: args.gpuPool.taints?.map((t) => ({
                        key: t.key,
                        value: t.value,
                        effect: t.effect as "NO_SCHEDULE" | "PREFER_NO_SCHEDULE" | "NO_EXECUTE",
                    })) ?? [
                        {
                            key: "nvidia.com/gpu",
                            value: "present",
                            effect: "NO_SCHEDULE" as const,
                        },
                    ],

                    metadata: {
                        "disable-legacy-endpoints": "true",
                    },

                    shieldedInstanceConfig: {
                        enableSecureBoot: true,
                        enableIntegrityMonitoring: true,
                    },
                },

                management: {
                    autoRepair: true,
                    autoUpgrade: true,
                },

                upgradeSettings: {
                    maxSurge: 1,
                    maxUnavailable: 0,
                },
            },
            { parent: this }
        );

        this.registerOutputs({
            cpuNodePool: this.cpuNodePool,
            gpuNodePool: this.gpuNodePool,
        });
    }
}

// =============================================================================
// GPU TYPES REFERENCE
// =============================================================================
// Common GPU types available in GKE and their characteristics:
//
// | GPU Type        | VRAM  | Use Case                    | Cost (on-demand) |
// |-----------------|-------|-----------------------------|--------------------|
// | nvidia-tesla-t4 | 16GB  | Inference, small training   | $0.35/hr          |
// | nvidia-l4       | 24GB  | Inference, medium training  | $0.70/hr          |
// | nvidia-tesla-a100 | 40GB | Large training             | $3.67/hr          |
// | nvidia-a100-80gb | 80GB | Very large models          | $4.00/hr          |
// | nvidia-h100-80gb | 80GB | Cutting edge training      | $5.00/hr          |
//
// Machine type requirements:
// - T4, L4: n1-standard-*, n2-standard-*
// - A100: a2-highgpu-*, a2-megagpu-*
// - H100: a3-highgpu-*
