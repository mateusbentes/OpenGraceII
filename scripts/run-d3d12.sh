#!/usr/bin/env bash
# Runs a D3D12 example under Wine, retrying on the lavapipe startup
# crash (the vkd3d-to-lavapipe device creation occasionally dies in the
# driver's instance enumeration before the engine's own code runs).
# Usage: ./run-d3d12.sh <example-name> [tries]
set -u
example="${1:?usage: run-d3d12.sh <example-name> [tries]}"
tries="${2:-4}"
exe="target/x86_64-pc-windows-gnu/debug/examples/${example}.exe"
if [ ! -f "$exe" ]; then
    echo "SKIP: $exe not built (cargo build --target x86_64-pc-windows-gnu --features d3d12 --examples)"
    exit 77
fi
for attempt in $(seq 1 "$tries"); do
    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json wine "$exe" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        exit 0
    fi
    echo "  (attempt $attempt exited $rc — the lavapipe startup flake; retrying)"
done
echo "FAIL: $example failed $tries times"
exit 1
