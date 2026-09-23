#!/bin/bash
# setup.sh: install build tools + CUDA toolkit + Rust, then build the binary.
# Linux only. Auto-detects the package manager and skips work that is already done.

set -e
set -o pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}==>${NC} $*"; }
warn()  { echo -e "${YELLOW}!!${NC} $*"; }
fail()  { echo -e "${RED}xx${NC} $*"; exit 1; }
ok()    { echo -e "${GREEN}OK${NC} $*"; }

[ "$(uname -s)" = "Linux" ] || fail "Linux only. On Windows use WSL with GPU passthrough; macOS is unsupported."
ARCH="$(uname -m)"

if command -v sudo >/dev/null 2>&1 && [ "$EUID" -ne 0 ]; then SUDO="sudo"; else SUDO=""; fi

DISTRO_ID=""; DISTRO_VER=""; DISTRO_LIKE=""
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    DISTRO_ID="${ID:-}"; DISTRO_VER="${VERSION_ID:-}"; DISTRO_LIKE="${ID_LIKE:-}"
fi

PKG_MGR=""
if   command -v apt-get >/dev/null 2>&1; then PKG_MGR=apt
elif command -v dnf     >/dev/null 2>&1; then PKG_MGR=dnf
elif command -v yum     >/dev/null 2>&1; then PKG_MGR=yum
elif command -v pacman  >/dev/null 2>&1; then PKG_MGR=pacman
elif command -v apk     >/dev/null 2>&1; then PKG_MGR=apk
else fail "No supported package manager (apt/dnf/yum/pacman/apk)."
fi
info "Detected: $DISTRO_ID $DISTRO_VER ($ARCH), package manager $PKG_MGR"

has_libnvrtc()   { ldconfig -p 2>/dev/null | grep -q 'libnvrtc\.so'; }
has_libcuda()    { ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1'; }
has_nvidia_smi() { command -v nvidia-smi >/dev/null 2>&1; }

# Step 1: build toolchain.
info "Installing build toolchain..."
case "$PKG_MGR" in
    apt)
        $SUDO apt-get update -qq
        $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y \
            build-essential pkg-config libssl-dev curl git ca-certificates
        ;;
    dnf)
        $SUDO dnf install -y --setopt=install_weak_deps=False \
            gcc gcc-c++ make pkgconf-pkg-config openssl-devel curl git ca-certificates
        ;;
    yum)
        $SUDO yum install -y \
            gcc gcc-c++ make pkgconfig openssl-devel curl git ca-certificates
        ;;
    pacman)
        $SUDO pacman -Sy --needed --noconfirm \
            base-devel openssl curl git ca-certificates
        ;;
    apk)
        $SUDO apk add build-base openssl-dev pkgconfig curl git ca-certificates
        ;;
esac
ok "Build toolchain ready."

# Step 2: NVIDIA driver check. Drivers ship libcuda.so.1 + nvidia-smi.
# We do not install the driver here because the right method varies wildly per host
# (datacenter image, gaming distro, headless server, container, WSL). Tell the user
# the canonical command for their distro and continue with the toolkit install.
if has_libcuda && has_nvidia_smi; then
    ok "NVIDIA driver present ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1))."
else
    warn "NVIDIA driver not detected. The binary will fail to run until you install it."
    case "$DISTRO_ID:$DISTRO_LIKE" in
        ubuntu:*|*:*ubuntu*|*:*debian*) warn "  Ubuntu/Debian: ${SUDO} ubuntu-drivers autoinstall  (or:  ${SUDO} apt-get install nvidia-driver-XXX)" ;;
        fedora:*)                       warn "  Fedora: enable RPM Fusion, then  ${SUDO} dnf install akmod-nvidia" ;;
        rhel:*|rocky:*|almalinux:*|centos:*) warn "  RHEL/Rocky/AlmaLinux: enable EPEL + the NVIDIA CUDA repo, then  ${SUDO} dnf install nvidia-driver" ;;
        arch:*|*:*arch*)                warn "  Arch: ${SUDO} pacman -S nvidia  (or nvidia-dkms for non-LTS kernels)" ;;
        *)                              warn "  See https://www.nvidia.com/Download/index.aspx" ;;
    esac
fi

# Step 3: CUDA toolkit (provides libnvrtc.so used by the binary at runtime).
if has_libnvrtc; then
    ok "CUDA toolkit already present."
else
    info "Installing CUDA toolkit..."
    case "$PKG_MGR" in
        apt)
            # nvidia-cuda-toolkit is in Ubuntu universe and Debian main. Provides CUDA
            # 11.x/12.x including libnvrtc, which is what the binary needs at runtime.
            $SUDO apt-get install -y nvidia-cuda-toolkit
            ;;
        dnf|yum)
            # NVIDIA distributes a network repo for RHEL/Fedora/Rocky/AlmaLinux. We drop
            # the .repo file directly with curl so we do not depend on dnf-plugins-core
            # or the dnf4-vs-dnf5 config-manager spelling.
            DISTRO_TAG=""
            case "$DISTRO_ID" in
                fedora) DISTRO_TAG="fedora${DISTRO_VER}" ;;
                rhel|rocky|almalinux|centos|ol)
                    DISTRO_TAG="rhel${DISTRO_VER%%.*}"
                    ;;
            esac
            if [ -n "$DISTRO_TAG" ]; then
                REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO_TAG}/${ARCH}/cuda-${DISTRO_TAG}.repo"
                if ! ls /etc/yum.repos.d/cuda*.repo >/dev/null 2>&1; then
                    info "Adding NVIDIA CUDA repo: $REPO_URL"
                    $SUDO curl -fsSL "$REPO_URL" -o /etc/yum.repos.d/cuda.repo \
                        || fail "Could not fetch $REPO_URL - check connectivity and distro tag."
                fi
            else
                warn "Unknown $DISTRO_ID $DISTRO_VER - add the NVIDIA CUDA repo manually."
            fi
            $SUDO $PKG_MGR clean expire-cache >/dev/null 2>&1 || true
            if ! $SUDO $PKG_MGR install -y cuda-toolkit; then
                $SUDO $PKG_MGR install -y cuda
            fi
            ;;
        pacman)
            $SUDO pacman -S --needed --noconfirm cuda
            ;;
        apk)
            fail "Alpine's apk does not ship CUDA. Install it manually from https://developer.nvidia.com/cuda-downloads, then re-run."
            ;;
    esac

    # If CUDA installed under /usr/local/cuda or /opt/cuda but libnvrtc still is not
    # discoverable by ldconfig, register its lib dir explicitly.
    if ! has_libnvrtc; then
        for cdir in /usr/local/cuda*/lib64 /opt/cuda/lib64 /usr/local/cuda*/targets/${ARCH}-linux/lib; do
            [ -d "$cdir" ] || continue
            if ls "$cdir"/libnvrtc.so* >/dev/null 2>&1; then
                info "Registering $cdir with ldconfig..."
                echo "$cdir" | $SUDO tee /etc/ld.so.conf.d/cuda.conf >/dev/null
                $SUDO ldconfig
                break
            fi
        done
    fi
    has_libnvrtc || fail "CUDA toolkit installed but libnvrtc.so is still not on the loader path."
    ok "CUDA toolkit ready."
fi

# Step 4: Rust toolchain.
if ! command -v cargo >/dev/null 2>&1; then
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null 2>&1 || fail "cargo not on PATH after rustup install."
ok "Rust toolchain: $(cargo --version)"

# Step 5: locate or clone the project, then build.
if [ ! -f "Cargo.toml" ] || ! grep -q "seedphrase_recovery" Cargo.toml 2>/dev/null; then
    info "Cloning RTX 3070 optimization fork..."
    CLONE_DIR="${SEEDPHRASE_CLONE_DIR:-btc-seedphrase-recovery-3070}"
    REPO_URL="${SEEDPHRASE_REPO_URL:-https://github.com/TnzGit/btc-seedphrase-recovery-3070.git}"
    REPO_BRANCH="${SEEDPHRASE_REPO_BRANCH:-opt/sm86-rtx3070-r2}"
    [ -d "$CLONE_DIR" ] && fail "Directory '$CLONE_DIR' already exists - cd into it and run ./setup.sh from inside."
    git clone --depth=1 --branch "$REPO_BRANCH" "$REPO_URL" "$CLONE_DIR"
    cd "$CLONE_DIR"
fi
PROJECT_DIR="$(pwd)"

info "Building (cargo build --release) in $PROJECT_DIR ..."
cargo build --release
BIN="$PROJECT_DIR/target/release/seedphrase_recovery"
[ -x "$BIN" ] || fail "Build claimed success but binary missing at $BIN"
ok "Built: $BIN"

# Step 6: detected GPUs.
if has_nvidia_smi && nvidia-smi -L >/dev/null 2>&1; then
    info "Detected GPU(s):"
    nvidia-smi -L | sed 's/^/    /'
fi

echo ""
ok "Setup complete."
echo ""
echo "  Run the tool:"
echo -e "    ${YELLOW}$BIN${NC}"
echo ""
echo "  Benchmark your GPU (~12s):"
echo -e "    ${YELLOW}$BIN --bench${NC}"
echo ""
