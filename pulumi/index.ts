import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";
import * as k8s from "@pulumi/kubernetes";

import { GkeCluster } from "./components/gke-cluster";
import { NodePools } from "./components/node-pools";
import { KubeRayOperator } from "./components/kuberay";

// =============================================================================
// CONFIGURATION
// =============================================================================
// Pulumi Config allows you to parameterize your infrastructure.
// Values come from Pulumi.<stack>.yaml files (e.g., Pulumi.dev.yaml)

const config = new pulumi.Config();
const gcpConfig = new pulumi.Config("gcp");

const clusterName = config.require("clusterName");
const region = gcpConfig.require("region");
const zone = gcpConfig.require("zone");

// Structured config for node pools - note the typing
interface NodePoolConfig {
    minNodes: number;
    maxNodes: number;
    machineType: string;
    preemptible: boolean;
    acceleratorType?: string;
    acceleratorCount?: number;
}

const cpuPoolConfig = config.requireObject<NodePoolConfig>("cpuPoolConfig");
const gpuPoolConfig = config.requireObject<NodePoolConfig>("gpuPoolConfig");

// =============================================================================
// NETWORKING (VPC)
// =============================================================================
// Every GKE cluster needs a VPC. We create a dedicated one for isolation.
//
// KEY CONCEPT: VPC-native clusters use alias IPs, which means:
// - Pods get IPs from a secondary range (not NAT'd)
// - Direct pod-to-pod communication across nodes
// - Required for private GKE clusters

const network = new gcp.compute.Network("rl-cluster-network", {
    name: `${clusterName}-network`,
    autoCreateSubnetworks: false,  // We'll create subnets explicitly
    description: "VPC for RL training cluster",
});

const subnet = new gcp.compute.Subnetwork("rl-cluster-subnet", {
    name: `${clusterName}-subnet`,
    network: network.id,
    region: region,
    ipCidrRange: "10.0.0.0/20",  // ~4096 IPs for nodes

    // Secondary ranges for pods and services (VPC-native cluster requirement)
    secondaryIpRanges: [
        {
            rangeName: "pods",
            ipCidrRange: "10.1.0.0/16",  // ~65k pod IPs
        },
        {
            rangeName: "services",
            ipCidrRange: "10.2.0.0/20",  // ~4096 service IPs
        },
    ],

    // Private Google Access allows nodes without external IPs to reach Google APIs
    privateIpGoogleAccess: true,
});

// =============================================================================
// GKE CLUSTER
// =============================================================================
// The GKE cluster is the Kubernetes control plane + default node pool.
//
// KEY CONCEPTS:
// - Control plane: API server, scheduler, controller-manager (managed by Google)
// - Node pools: Groups of worker VMs with identical config
// - We create a minimal default pool, then add specialized pools

const gkeCluster = new GkeCluster("rl-cluster", {
    name: clusterName,
    location: zone,  // Zonal cluster (cheaper) vs regional (HA)
    network: network.name,
    subnetwork: subnet.name,
    podsRangeName: "pods",
    servicesRangeName: "services",

    // Cluster features
    enableWorkloadIdentity: true,   // Secure pod-to-GCP-service auth
    releaseChannel: "REGULAR",       // Auto-upgrades with stability
});

// =============================================================================
// NODE POOLS
// =============================================================================
// Separate pools for different workload types.
//
// WHY SEPARATE POOLS?
// 1. Different machine types (CPU vs GPU)
// 2. Different scaling policies
// 3. Cost optimization (preemptible for batch, on-demand for services)
// 4. Taints to control pod placement

const nodePools = new NodePools("rl-node-pools", {
    clusterName: gkeCluster.cluster.name,
    location: zone,

    cpuPool: {
        name: "cpu-workers",
        ...cpuPoolConfig,
        // Labels help with pod scheduling via nodeSelector
        labels: {
            "workload-type": "cpu",
            "ray.io/node-type": "worker",
        },
    },

    gpuPool: {
        name: "gpu-workers",
        ...gpuPoolConfig,
        labels: {
            "workload-type": "gpu",
            "ray.io/node-type": "worker",
        },
        // Taints prevent non-GPU workloads from landing here
        taints: [{
            key: "nvidia.com/gpu",
            value: "present",
            effect: "NO_SCHEDULE",
        }],
    },
});

// =============================================================================
// KUBERNETES PROVIDER
// =============================================================================
// To deploy K8s resources (like KubeRay), we need a K8s provider
// configured with credentials from our GKE cluster.

const k8sProvider = new k8s.Provider("gke-k8s-provider", {
    kubeconfig: gkeCluster.kubeconfig,
});

// =============================================================================
// KUBERAY OPERATOR
// =============================================================================
// KubeRay is a K8s operator that manages Ray clusters.
//
// KEY CONCEPT - Operators:
// Operators extend K8s with custom resources. Instead of managing
// individual pods, you declare a "RayCluster" and the operator
// creates/manages all the underlying resources.

const kuberay = new KubeRayOperator("kuberay", {
    namespace: "ray-system",
}, { provider: k8sProvider, dependsOn: [nodePools] });

// =============================================================================
// OUTPUTS
// =============================================================================
// Exports make values available to:
// 1. Other Pulumi stacks (pulumi stack output)
// 2. CI/CD pipelines
// 3. Your terminal after `pulumi up`

export const clusterEndpoint = gkeCluster.cluster.endpoint;
export const clusterCaCertificate = pulumi.secret(
    gkeCluster.cluster.masterAuth.clusterCaCertificate
);

// Command to get kubectl credentials
export const kubectlCommand = pulumi.interpolate`gcloud container clusters get-credentials ${clusterName} --zone ${zone}`;

// Useful info for debugging
export const nodePoolInfo = {
    cpuPool: {
        name: "cpu-workers",
        machineType: cpuPoolConfig.machineType,
        scaling: `${cpuPoolConfig.minNodes}-${cpuPoolConfig.maxNodes}`,
    },
    gpuPool: {
        name: "gpu-workers",
        machineType: gpuPoolConfig.machineType,
        gpu: gpuPoolConfig.acceleratorType,
        scaling: `${gpuPoolConfig.minNodes}-${gpuPoolConfig.maxNodes}`,
    },
};
