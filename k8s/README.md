# AgentInvest Kubernetes Deployment Configuration

## 📁 Clean Deployment Structure

This directory contains the streamlined Kubernetes deployment configuration for AgentInvest with all recent fixes incorporated.

### 🗂️ File Organization

```
webapp/k8s/
├── README.md                           # This documentation
├── agentinvest-complete.yaml          # Complete infrastructure deployment
├── enhanced-backend-deployment.yaml   # Backend with analytics & WebSocket
├── enhanced-frontend-deployment.yaml  # Frontend with Chart.js fixes
├── enhanced-nginx-proxy.yaml         # Nginx proxy with WebSocket support
├── namespace.yaml                     # Namespace definition
├── secrets.yaml                       # Application secrets
├── reports-pvc.yaml                   # Reports storage
├── configmap.yaml                     # Legacy config (kept for reference)
├── postgresql-deployment-production.yaml  # PostgreSQL production config
└── redis-deployment-production.yaml   # Redis production config
```

## 🚀 Deployment Components

### **Core Application (Enhanced)**
- **`enhanced-backend-deployment.yaml`**: Backend API with analytics endpoints and WebSocket support
- **`enhanced-frontend-deployment.yaml`**: React frontend with Chart.js date adapter fixes
- **`enhanced-nginx-proxy.yaml`**: Nginx reverse proxy with WebSocket routing

### **Infrastructure (Complete)**
- **`agentinvest-complete.yaml`**: All-in-one infrastructure deployment including:
  - Namespace and ConfigMaps
  - PostgreSQL database
  - Redis cache
  - Persistent Volume Claims
  - Services

### **Configuration**
- **`namespace.yaml`**: functorhk namespace
- **`secrets.yaml`**: Database passwords and API keys
- **`reports-pvc.yaml`**: Persistent storage for generated reports
- **`rabbitmq-deployment.yaml`**: RabbitMQ message broker StatefulSet ⭐ **NEW**
- **`rabbitmq-init-job.yaml`**: AgentInvest queue and exchange initialization ⭐ **NEW**
- **`rabbitmq-management-service.yaml`**: RabbitMQ management UI and external access ⭐ **NEW**
- **`rabbitmq-monitoring.yaml`**: Health checks and Prometheus monitoring ⭐ **NEW**

## ✅ Fixes Incorporated

### **1. Chart.js Date Adapter Fix**
- **File**: `enhanced-frontend-deployment.yaml`
- **Fix**: Includes `chartjs-adapter-date-fns@^3.0.0` and `date-fns@^3.6.0`
- **Result**: Resolves "This method is not implemented" Chart.js errors

### **2. Analytics API Endpoints**
- **File**: `enhanced-backend-deployment.yaml`
- **Fix**: Backend image includes complete analytics router
- **Endpoints**:
  - `/api/feedback/analytics/overview`
  - `/api/feedback/analytics/trends?days=7`
  - `/api/feedback/analytics/reports?limit=10`
  - `/api/feedback/analytics/recent?limit=20`

### **3. WebSocket Support**
- **Files**: `enhanced-backend-deployment.yaml`, `enhanced-nginx-proxy.yaml`
- **Fix**: Complete WebSocket implementation with proxy support
- **Endpoint**: `/ws/reports/{report_id}` for real-time updates

### **4. Enhanced Kubernetes Features**
- **Health Checks**: Liveness, readiness, and startup probes
- **Resource Limits**: Memory and CPU constraints
- **Security**: Non-root containers, read-only filesystems
- **Rolling Updates**: Zero-downtime deployments

## 🔧 Deployment Methods

### **Option 1: Automated Deployment (Recommended)**
```bash
# Use the enhanced deployment script
./deploy-enhanced.sh
```

### **Option 2: Manual Deployment**
```bash
# Set kubeconfig
export KUBECONFIG="/path/to/kubeconfig.yaml"

# Deploy infrastructure
kubectl apply -f webapp/k8s/namespace.yaml
kubectl apply -f webapp/k8s/secrets.yaml
kubectl apply -f webapp/k8s/agentinvest-complete.yaml

# Deploy RabbitMQ message queue (NEW)
kubectl apply -f webapp/k8s/rabbitmq-deployment.yaml
kubectl wait --for=condition=ready pod -l app=rabbitmq -n agentinvest --timeout=300s
kubectl apply -f webapp/k8s/rabbitmq-init-job.yaml
kubectl apply -f webapp/k8s/rabbitmq-management-service.yaml
kubectl apply -f webapp/k8s/rabbitmq-monitoring.yaml

# Deploy enhanced applications
kubectl apply -f webapp/k8s/enhanced-backend-deployment.yaml
kubectl apply -f webapp/k8s/enhanced-frontend-deployment.yaml
kubectl apply -f webapp/k8s/enhanced-nginx-proxy.yaml

# Wait for deployments
kubectl wait --for=condition=available --timeout=600s deployment/webapp-backend -n functorhk
kubectl wait --for=condition=available --timeout=600s deployment/webapp-frontend -n functorhk
kubectl wait --for=condition=available --timeout=600s deployment/nginx-proxy -n functorhk
```

### **Option 3: Component-by-Component**
```bash
# Infrastructure only
kubectl apply -f webapp/k8s/agentinvest-complete.yaml

# Backend only
kubectl apply -f webapp/k8s/enhanced-backend-deployment.yaml

# Frontend only
kubectl apply -f webapp/k8s/enhanced-frontend-deployment.yaml

# Proxy only
kubectl apply -f webapp/k8s/enhanced-nginx-proxy.yaml
```

## 🧹 Cleanup History

### **Removed Files (Outdated)**
- `frontend-deployment.yaml` → Replaced by `enhanced-frontend-deployment.yaml`
- `backend-deployment.yaml` → Replaced by `enhanced-backend-deployment.yaml`
- `backend-deployment-fixed.yaml` → Consolidated into enhanced version
- `analytics-fix-configmap.yaml` → Integrated into complete manifest
- `ingress-fixed.yaml` → Replaced by nginx proxy
- `redis-deployment.yaml` → Replaced by production version
- `redis-deployment-current.yaml` → Replaced by production version
- `ingress/` directory → Entire directory removed (using nginx proxy instead)

### **Consolidated Components**
- **ConfigMaps**: All configuration consolidated into `agentinvest-complete.yaml`
- **Services**: Database and cache services included in complete manifest
- **PVCs**: All persistent volume claims in single file
- **Deployments**: Enhanced versions with all fixes included

## 📊 Application Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │   Frontend      │    │   Backend       │
│   Port: 30084   │◄──►│   React App     │◄──►│   FastAPI       │
│   WebSocket     │    │   Chart.js Fix  │    │   Analytics API │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐                          ┌─────────────────┐
│   PostgreSQL    │                          │     Redis       │
│   Database      │                          │     Cache       │
│   Port: 5432    │                          │   Port: 6379    │
└─────────────────┘                          └─────────────────┘
```

## 🎯 Success Criteria

After deployment, verify:

✅ **All pods running**: `kubectl get pods -n functorhk`
✅ **Frontend accessible**: http://agentinvest.applenova.store:30084/
✅ **Backend health**: http://agentinvest.applenova.store:30084/api/health
✅ **Analytics endpoints**: All 4 analytics endpoints return HTTP 200
✅ **No Chart.js errors**: Browser console shows no date adapter errors
✅ **WebSocket routing**: `/ws/` endpoints properly routed (404 expected)

## 🔍 Troubleshooting

### **Common Issues**
1. **Pod not starting**: Check resource limits and node capacity
2. **Service not accessible**: Verify service endpoints and port configuration
3. **Chart.js errors**: Ensure frontend image includes date adapter dependencies
4. **Analytics 404**: Verify backend image tag includes analytics endpoints

### **Debugging Commands**
```bash
# Check pod status
kubectl get pods -n functorhk -o wide

# View logs
kubectl logs -l app=webapp-backend -n functorhk --tail=100
kubectl logs -l app=webapp-frontend -n functorhk --tail=100
kubectl logs -l app=nginx-proxy -n functorhk --tail=100

# Check services
kubectl get services -n functorhk
kubectl get endpoints -n functorhk

# Describe deployments
kubectl describe deployment webapp-backend -n functorhk
```

## 📝 Maintenance

### **Updating Images**
```bash
# Update backend
kubectl set image deployment/webapp-backend backend=functorhk/webapp-backend:new-tag -n functorhk

# Update frontend
kubectl set image deployment/webapp-frontend frontend=functorhk/webapp-frontend:new-tag -n functorhk
```

### **Scaling**
```bash
# Scale backend
kubectl scale deployment webapp-backend --replicas=3 -n functorhk

# Scale frontend
kubectl scale deployment webapp-frontend --replicas=3 -n functorhk
```

This clean deployment structure ensures all fixes are properly incorporated and eliminates conflicts between old and new configurations.
