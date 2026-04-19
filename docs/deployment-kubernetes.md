# Kubernetes Deployment Guide

The manifests in [`deploy/kubernetes`](/Users/dileepuk/Desktop/Developer/llm-gateway/deploy/kubernetes) split the platform into API and worker deployments.

## Apply order

1. `namespace.yaml`
2. `configmap.yaml`
3. your real secret based on `secret.example.yaml`
4. `gateway-deployment.yaml`
5. `worker-deployment.yaml`
6. `service.yaml`
7. `ingress.yaml`
8. `hpa.yaml`
9. `networkpolicy.yaml`

## Operational notes

- API pods serve `/`, `/admin`, and the JSON APIs.
- Worker pods run the async runtime without binding HTTP.
- Both pods must share the same `PLATFORM_SIGNING_SECRET`.
- Mount persistent artifact storage in production instead of `emptyDir`.
- Use staging and production namespaces with separate secrets and budgets.
