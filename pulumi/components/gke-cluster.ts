import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";

// =============================================================================
// GKE CLUSTER COMPONENT
// =============================================================================
// This is a Pulumi ComponentResource - a reusable abstraction.
//
// KEY PULUMI CONCEPT - Component Resources:
// - Group related resources into a logical unit
// - Encapsulate best practices
// - Provide a clean interface (inputs/outputs)
// - Show as a single item in Pulumi's resource tree
//
// Think of it like a class that creates infrastructure.

export interface GkeClusterArgs {
    name: string;
    location: string;           // Zone or region
    network: pulumi.Input<string>;
    subnetwork: pulumi.Input<string>;
    podsRangeName: string;
    servicesRangeName: string;
    releaseChannel?: string;
    enableWorkloadIdentity?: boolean;
}

export class GkeCluster extends pulumi.ComponentResource {
    public readonly cluster: gcp.container.Cluster;
    public readonly kubeconfig: pulumi.Output<string>;

    constructor(
        name: string,
        args: GkeClusterArgs,
        opts?: pulumi.ComponentResourceOptions
    ) {
        // Register this component with Pulumi
        super("custom:gke:GkeCluster", name, {}, opts);

        // =====================================================================
        // SERVICE ACCOUNT
        // =====================================================================
        // Nodes need a service account to interact with GCP services.
        // Best practice: Create a dedicated SA with minimal permissions.

        const nodeServiceAccount = new gcp.serviceaccount.Account(
            `${name}-node-sa`,
            {
                accountId: `${args.name}-nodes`,
                displayName: "GKE Node Service Account",
            },
            { parent: this }
        );

        // Grant necessary roles to the node service account
        // These are the minimal roles for GKE nodes
        const roles = [
            "roles/logging.logWriter",           // Write logs to Cloud Logging
            "roles/monitoring.metricWriter",      // Write metrics to Cloud Monitoring
            "roles/monitoring.viewer",            // Read monitoring data
            "roles/artifactregistry.reader",      // Pull container images
        ];

        roles.forEach((role, index) => {
            new gcp.projects.IAMMember(
                `${name}-node-sa-role-${index}`,
                {
                    project: gcp.config.project!,
                    role: role,
                    member: pulumi.interpolate`serviceAccount:${nodeServiceAccount.email}`,
                },
                { parent: this }
            );
        });

        // =====================================================================
        // GKE CLUSTER
        // =====================================================================
        // The cluster resource creates the K8s control plane and a default pool.
        // We remove the default pool immediately (removeDefaultNodePool: true)
        // and create our own specialized pools.

        this.cluster = new gcp.container.Cluster(
            `${name}-cluster`,
            {
                name: args.name,
                location: args.location,

                // Networking
                network: args.network,
                subnetwork: args.subnetwork,

                // VPC-native cluster configuration
                // This enables alias IPs so pods get routable IPs
                ipAllocationPolicy: {
                    clusterSecondaryRangeName: args.podsRangeName,
                    servicesSecondaryRangeName: args.servicesRangeName,
                },

                // Remove default node pool - we'll create our own
                // This is a common pattern for production clusters
                removeDefaultNodePool: true,
                initialNodeCount: 1,  // Required but immediately removed

                // Release channel for automatic upgrades
                // RAPID: Newest features, less stable
                // REGULAR: Balance of features and stability
                // STABLE: Most stable, features delayed
                releaseChannel: {
                    channel: args.releaseChannel ?? "REGULAR",
                },

                // Workload Identity: Secure way for pods to access GCP services
                // Instead of mounting service account keys, pods assume identities
                workloadIdentityConfig: args.enableWorkloadIdentity
                    ? {
                          workloadPool: pulumi.interpolate`${gcp.config.project}.svc.id.goog`,
                      }
                    : undefined,

                // Cluster add-ons
                addonsConfig: {
                    // HTTP load balancing for Ingress resources
                    httpLoadBalancing: { disabled: false },

                    // Horizontal Pod Autoscaler
                    horizontalPodAutoscaling: { disabled: false },

                    // GCE Persistent Disk CSI Driver
                    gcePersistentDiskCsiDriverConfig: { enabled: true },

                    // GCS FUSE CSI Driver - mount GCS buckets as filesystems
                    // Useful for large datasets
                    gcsFuseCsiDriverConfig: { enabled: true },
                },

                // Logging and monitoring
                loggingConfig: {
                    enableComponents: [
                        "SYSTEM_COMPONENTS",
                        "WORKLOADS",
                    ],
                },
                monitoringConfig: {
                    enableComponents: [
                        "SYSTEM_COMPONENTS",
                    ],
                    managedPrometheus: {
                        enabled: true,  // Google-managed Prometheus
                    },
                },

                // Network policy enforcement (for pod-to-pod firewall rules)
                networkPolicy: {
                    enabled: true,
                    provider: "CALICO",
                },

                // Maintenance window - when GKE can auto-upgrade
                // Set to off-hours for your region
                maintenancePolicy: {
                    dailyMaintenanceWindow: {
                        startTime: "03:00",  // 3 AM UTC
                    },
                },

                // Default labels applied to all resources
                resourceLabels: {
                    environment: "dev",
                    managed_by: "pulumi",
                    purpose: "rl-training",
                },
            },
            { parent: this }
        );

        // =====================================================================
        // KUBECONFIG GENERATION
        // =====================================================================
        // Generate a kubeconfig that can be used by kubectl and Pulumi's K8s provider

        this.kubeconfig = pulumi
            .all([
                this.cluster.name,
                this.cluster.endpoint,
                this.cluster.masterAuth,
            ])
            .apply(([name, endpoint, masterAuth]) => {
                const context = `gke_${gcp.config.project}_${args.location}_${name}`;
                return `apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: ${masterAuth.clusterCaCertificate}
    server: https://${endpoint}
  name: ${context}
contexts:
- context:
    cluster: ${context}
    user: ${context}
  name: ${context}
current-context: ${context}
kind: Config
preferences: {}
users:
- name: ${context}
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1beta1
      command: gke-gcloud-auth-plugin
      installHint: Install gke-gcloud-auth-plugin for kubectl auth
      provideClusterInfo: true
`;
            });

        // Register outputs for this component
        this.registerOutputs({
            cluster: this.cluster,
            kubeconfig: this.kubeconfig,
        });
    }
}
