#!/bin/bash

# Production deployment script for Tiny-VLM
# This script automates the deployment process with comprehensive validation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="/tmp/tiny-vlm-deploy-$(date +%s).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}INFO${NC}: $1"
}

log_success() {
    log "${GREEN}SUCCESS${NC}: $1"
}

log_warning() {
    log "${YELLOW}WARNING${NC}: $1"
}

log_error() {
    log "${RED}ERROR${NC}: $1"
}

# Error handler
error_handler() {
    log_error "Deployment failed at line $1"
    log_error "Check the log file: $LOG_FILE"
    exit 1
}

trap 'error_handler $LINENO' ERR

# Print banner
print_banner() {
    echo "
 ████████╗██╗███╗   ██╗██╗   ██╗      ██╗   ██╗██╗     ███╗   ███╗
 ╚══██╔══╝██║████╗  ██║╚██╗ ██╔╝      ██║   ██║██║     ████╗ ████║
    ██║   ██║██╔██╗ ██║ ╚████╔╝ █████╗██║   ██║██║     ██╔████╔██║
    ██║   ██║██║╚██╗██║  ╚██╔╝  ╚════╝╚██╗ ██╔╝██║     ██║╚██╔╝██║
    ██║   ██║██║ ╚████║   ██║          ╚████╔╝ ███████╗██║ ╚═╝ ██║
    ╚═╝   ╚═╝╚═╝  ╚═══╝   ╚═╝           ╚═══╝  ╚══════╝╚═╝     ╚═╝
    
    🚀 Production Deployment Script v1.0.0
    🏢 Terragon Labs - Autonomous SDLC Implementation
    
"
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    if ! command -v docker &> /dev/null; then
        missing_tools+("docker")
    fi
    
    if ! command -v kubectl &> /dev/null; then
        missing_tools+("kubectl")
    fi
    
    if ! command -v helm &> /dev/null; then
        log_warning "Helm not found - Kubernetes deployment may be limited"
    fi
    
    if ! command -v cargo &> /dev/null; then
        missing_tools+("cargo")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "All prerequisites validated"
}

# Build and test the application
build_and_test() {
    log_info "Building and testing the application..."
    
    cd "$PROJECT_ROOT"
    
    # Clean previous builds
    log_info "Cleaning previous builds..."
    cargo clean
    
    # Build in release mode
    log_info "Building in release mode with optimizations..."
    cargo build --release --features std,simd
    
    # Run comprehensive tests
    log_info "Running comprehensive test suite..."
    cargo test --release --features std
    
    # Security audit
    log_info "Running security audit..."
    if command -v cargo-audit &> /dev/null; then
        cargo audit
    else
        log_warning "cargo-audit not installed - skipping security audit"
    fi
    
    # Performance benchmarks
    log_info "Running performance benchmarks..."
    cargo bench --features std
    
    log_success "Build and tests completed successfully"
}

# Build Docker images
build_docker_images() {
    log_info "Building production Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build multi-arch image
    log_info "Building multi-architecture Docker image..."
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        -f deploy/docker/Dockerfile.production \
        -t tiny-vlm:latest \
        -t tiny-vlm:v1.0.0 \
        --push \
        .
    
    # Security scan
    log_info "Scanning Docker image for vulnerabilities..."
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL tiny-vlm:latest
    else
        log_warning "Trivy not installed - skipping security scan"
    fi
    
    log_success "Docker images built successfully"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Validate cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."
    kubectl apply -f "$PROJECT_ROOT/deploy/kubernetes/production.yaml"
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/tiny-vlm -n tiny-vlm-production
    
    # Verify deployment
    log_info "Verifying deployment..."
    kubectl get pods -n tiny-vlm-production -l app=tiny-vlm
    
    # Get service URLs
    log_info "Getting service information..."
    kubectl get services -n tiny-vlm-production
    kubectl get ingress -n tiny-vlm-production
    
    log_success "Kubernetes deployment completed"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT/deploy/docker"
    
    # Create monitoring directories
    mkdir -p monitoring/grafana/{dashboards,datasources}
    mkdir -p letsencrypt
    
    # Create Prometheus config
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'tiny-vlm'
    static_configs:
      - targets: ['tiny-vlm:9090']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
    
    # Deploy stack
    log_info "Starting production stack..."
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Health checks
    log_info "Performing health checks..."
    for i in {1..30}; do
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Health check passed"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Health check failed after 30 attempts"
            exit 1
        fi
        sleep 2
    done
    
    log_success "Docker Compose deployment completed"
}

# Run deployment validation
validate_deployment() {
    log_info "Running deployment validation..."
    
    # Test API endpoints
    log_info "Testing API endpoints..."
    
    if command -v curl &> /dev/null; then
        # Health check
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Health check endpoint working"
        else
            log_error "Health check endpoint failed"
        fi
        
        # Metrics endpoint
        if curl -f http://localhost:9090/metrics &> /dev/null; then
            log_success "Metrics endpoint working"
        else
            log_warning "Metrics endpoint not accessible"
        fi
    fi
    
    # Load testing (simple)
    log_info "Running basic load test..."
    if command -v ab &> /dev/null; then
        ab -n 100 -c 10 http://localhost:8080/health
    else
        log_warning "Apache Bench not installed - skipping load test"
    fi
    
    log_success "Deployment validation completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup operations here
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    print_banner
    
    local deployment_type=${1:-"docker-compose"}
    
    log_info "Starting Tiny-VLM production deployment"
    log_info "Deployment type: $deployment_type"
    log_info "Log file: $LOG_FILE"
    
    validate_prerequisites
    build_and_test
    build_docker_images
    
    case $deployment_type in
        "kubernetes"|"k8s")
            deploy_kubernetes
            ;;
        "docker-compose"|"compose")
            deploy_docker_compose
            ;;
        "both")
            deploy_docker_compose
            log_info "Docker Compose deployment completed, starting Kubernetes deployment..."
            deploy_kubernetes
            ;;
        *)
            log_error "Unknown deployment type: $deployment_type"
            log_error "Valid options: kubernetes, docker-compose, both"
            exit 1
            ;;
    esac
    
    validate_deployment
    cleanup
    
    log_success "🎉 Tiny-VLM production deployment completed successfully!"
    log_info "📋 Deployment summary logged to: $LOG_FILE"
    
    # Print access information
    echo "
    🌐 ACCESS INFORMATION:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    📡 API Endpoint: http://localhost:8080
    📊 Metrics:     http://localhost:9090/metrics
    💹 Grafana:     http://localhost:3000 (admin/secure_admin_password_change_me)
    📈 Prometheus:  http://localhost:9091
    
    🔍 Health Check: curl http://localhost:8080/health
    📋 Service Logs: docker-compose -f deploy/docker/docker-compose.production.yml logs -f
    
    🏁 Production deployment completed successfully!
    "
}

# Handle script arguments
case "${1:-help}" in
    "kubernetes"|"k8s"|"docker-compose"|"compose"|"both")
        main "$1"
        ;;
    "help"|"-h"|"--help")
        echo "
Usage: $0 [deployment-type]

Deployment Types:
  kubernetes     - Deploy to Kubernetes cluster
  docker-compose - Deploy using Docker Compose (default)
  both          - Deploy to both platforms
  help          - Show this help message

Examples:
  $0                    # Deploy using Docker Compose
  $0 docker-compose     # Deploy using Docker Compose
  $0 kubernetes         # Deploy to Kubernetes
  $0 both              # Deploy to both platforms
"
        ;;
    *)
        log_error "Unknown command: $1"
        log_error "Use '$0 help' for usage information"
        exit 1
        ;;
esac