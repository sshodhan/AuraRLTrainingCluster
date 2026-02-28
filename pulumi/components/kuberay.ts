import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";

// =============================================================================
// KUBERAY OPERATOR COMPONENT
// =============================================================================
// KubeRay is a Kubernetes operator for managing Ray clusters.
//
// KEY CONCEPT - Kubernetes Operators:
// Operators are software extensions to K8s that use custom resources
// to manage applications. Instead of managing individual pods/services,
// you declare the desired state of your application (e.g., "I want a
// Ray cluster with 3 workers") and the operator makes it happen.
//
// How KubeRay works:
// 1. You create a RayCluster custom resource
// 2. KubeRay controller watches for RayCluster resources
// 3. Controller creates/manages pods, services, autoscaling
// 4. If a worker dies, controller replaces it automatically

export interface KubeRayOperatorArgs {
    namespace: string;
    version?: string;
}

export class KubeRayOperator extends pulumi.ComponentResource {
    public readonly namespace: k8s.core.v1.Namespace;
    public readonly operatorDeployment: k8s.helm.v3.Release;

    constructor(
        name: string,
        args: KubeRayOperatorArgs,
        opts?: pulumi.ComponentResourceOptions
    ) {
        super("custom:kuberay:Operator", name, {}, opts);

        const version = args.version ?? "1.1.0";

        // =====================================================================
        // NAMESPACE
        // =====================================================================
        // Create a dedicated namespace for Ray system components.
        // Namespaces provide:
        // - Logical isolation
        // - Resource quotas
        // - RBAC boundaries

        this.namespace = new k8s.core.v1.Namespace(
            `${name}-namespace`,
            {
                metadata: {
                    name: args.namespace,
                    labels: {
                        "app.kubernetes.io/name": "kuberay",
                        "app.kubernetes.io/component": "operator",
                    },
                },
            },
            { parent: this, ...opts }
        );

        // =====================================================================
        // KUBERAY OPERATOR (via Helm)
        // =====================================================================
        // Helm is a package manager for Kubernetes.
        //
        // KEY CONCEPT - Helm:
        // - Charts: Packages of pre-configured K8s resources
        // - Values: Configuration that customizes the chart
        // - Release: An installed instance of a chart
        //
        // We use the official KubeRay Helm chart to install the operator.

        this.operatorDeployment = new k8s.helm.v3.Release(
            `${name}-operator`,
            {
                name: "kuberay-operator",
                namespace: this.namespace.metadata.name,
                chart: "kuberay-operator",
                version: version,
                repositoryOpts: {
                    repo: "https://ray-project.github.io/kuberay-helm/",
                },

                // Helm values to customize the operator
                values: {
                    // Operator replica count
                    replicaCount: 1,

                    // Resource requests/limits for the operator itself
                    resources: {
                        requests: {
                            cpu: "100m",
                            memory: "256Mi",
                        },
                        limits: {
                            cpu: "500m",
                            memory: "512Mi",
                        },
                    },

                    // Watch all namespaces for RayCluster resources
                    // Set to specific namespace list if you want isolation
                    watchNamespace: "",

                    // Leader election for HA (if running multiple replicas)
                    leaderElection: {
                        enabled: true,
                    },

                    // Batch scheduler integration (e.g., Volcano, Yunikorn)
                    // Useful for gang scheduling in distributed training
                    batchScheduler: {
                        enabled: false,
                    },
                },
            },
            { parent: this, dependsOn: [this.namespace], ...opts }
        );

        this.registerOutputs({
            namespace: this.namespace,
            operatorDeployment: this.operatorDeployment,
        });
    }
}

// =============================================================================
// RAY CLUSTER CUSTOM RESOURCE EXAMPLE
// =============================================================================
// This shows the structure of a RayCluster resource.
// You would create this after the operator is installed.

export interface RayClusterArgs {
    name: string;
    namespace: string;
    rayVersion?: string;
    headCpus?: string;
    headMemory?: string;
    workerReplicas?: number;
    workerCpus?: string;
    workerMemory?: string;
    workerGpus?: number;
}

export class RayCluster extends pulumi.ComponentResource {
    public readonly cluster: k8s.apiextensions.CustomResource;

    constructor(
        name: string,
        args: RayClusterArgs,
        opts?: pulumi.ComponentResourceOptions
    ) {
        super("custom:kuberay:RayCluster", name, {}, opts);

        const rayVersion = args.rayVersion ?? "2.9.0";

        // =====================================================================
        // RAY CLUSTER RESOURCE
        // =====================================================================
        // This is a Custom Resource defined by the KubeRay CRD.
        //
        // Structure:
        // - headGroupSpec: Configuration for the Ray head node
        // - workerGroupSpecs: Array of worker group configurations
        //
        // The operator watches for these resources and creates the
        // corresponding pods, services, and autoscaling rules.

        this.cluster = new k8s.apiextensions.CustomResource(
            `${name}-cluster`,
            {
                apiVersion: "ray.io/v1",
                kind: "RayCluster",
                metadata: {
                    name: args.name,
                    namespace: args.namespace,
                },
                spec: {
                    rayVersion: rayVersion,

                    // Enable autoscaling
                    enableInTreeAutoscaling: true,

                    // Head node configuration
                    headGroupSpec: {
                        rayStartParams: {
                            "dashboard-host": "0.0.0.0",
                            "num-cpus": "0",  // Head doesn't run tasks
                        },
                        template: {
                            spec: {
                                containers: [
                                    {
                                        name: "ray-head",
                                        image: `rayproject/ray:${rayVersion}`,
                                        resources: {
                                            requests: {
                                                cpu: args.headCpus ?? "2",
                                                memory: args.headMemory ?? "4Gi",
                                            },
                                            limits: {
                                                cpu: args.headCpus ?? "2",
                                                memory: args.headMemory ?? "4Gi",
                                            },
                                        },
                                        ports: [
                                            { containerPort: 6379, name: "gcs" },
                                            { containerPort: 8265, name: "dashboard" },
                                            { containerPort: 10001, name: "client" },
                                        ],
                                    },
                                ],
                            },
                        },
                    },

                    // Worker group(s) configuration
                    workerGroupSpecs: [
                        {
                            groupName: "gpu-workers",
                            replicas: args.workerReplicas ?? 1,
                            minReplicas: 0,
                            maxReplicas: 10,
                            rayStartParams: {},
                            template: {
                                spec: {
                                    containers: [
                                        {
                                            name: "ray-worker",
                                            image: `rayproject/ray:${rayVersion}-gpu`,
                                            resources: {
                                                requests: {
                                                    cpu: args.workerCpus ?? "4",
                                                    memory: args.workerMemory ?? "8Gi",
                                                    "nvidia.com/gpu": args.workerGpus ?? 1,
                                                },
                                                limits: {
                                                    cpu: args.workerCpus ?? "4",
                                                    memory: args.workerMemory ?? "8Gi",
                                                    "nvidia.com/gpu": args.workerGpus ?? 1,
                                                },
                                            },
                                        },
                                    ],
                                    // Toleration to run on tainted GPU nodes
                                    tolerations: [
                                        {
                                            key: "nvidia.com/gpu",
                                            operator: "Exists",
                                            effect: "NoSchedule",
                                        },
                                    ],
                                    // Prefer GPU nodes
                                    nodeSelector: {
                                        "gpu-type": "nvidia-tesla-t4",
                                    },
                                },
                            },
                        },
                    ],
                },
            },
            { parent: this, ...opts }
        );

        this.registerOutputs({
            cluster: this.cluster,
        });
    }
}
