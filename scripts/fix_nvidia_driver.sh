#!/usr/bin/env bash
# Restore the GTX 1660 Ti driver by DOWNGRADING THE KERNEL, not the driver.
#
# WHY THIS SHAPE. `nvidia 570.86.16-7` is installed and its module is built for
# kernel 6.13.4-arch1-1, but the running kernel is 7.1.9-arch1-2, so no nvidia.ko
# exists for it and nvidia-smi cannot reach the driver. `pacman -S nvidia-dkms`
# would pull the CURRENT nvidia release and risks losing Turing support, which
# this project's hardware notes warn about. Reinstalling the cached 6.13.4 kernel
# touches NO nvidia package at all -- the already-built, already-working module is
# simply given back the kernel it targets.
#
# VERIFIED BEFORE WRITING THIS:
#   * /usr/lib/modules/6.13.4-arch1-1/extramodules/nvidia.ko.zst exists and
#     `modinfo` reports version 570.86.16.
#   * linux-6.13.4.arch1-1 AND linux-7.1.9.arch1-2 are both in the local pacman
#     cache, so the revert never needs network.
#   * WiFi does NOT depend on DKMS: wlp4s0f3u3 uses the IN-TREE `rtl8192cu`
#     driver, /var/lib/dkms is empty, and no 8821cu module is loaded. So
#     rtl8821cu-dkms-git is inert and nothing needs rebuilding. No linux-headers
#     are required either, because the nvidia module is prebuilt rather than DKMS.
#   * nvidia is ALREADY in pacman.conf IgnorePkg; `linux` is NOT, which is how the
#     kernel moved out from under the driver. Step 4 fixes that.
#
# RESIDUAL RISK, stated honestly: 6.13.4 is ~18 months older than the current
# userspace. Nothing here depends on new kernel syscalls as far as could be
# checked, but an old kernel under new userspace is not a zero-risk combination.
# The revert at the bottom needs no network, and GRUB keeps an entry either way.
#
# Run with: sudo bash scripts/fix_nvidia_driver.sh
set -euo pipefail

KERNEL_PKG=/var/cache/pacman/pkg/linux-6.13.4.arch1-1-x86_64.pkg.tar.zst
REVERT_PKG=/var/cache/pacman/pkg/linux-7.1.9.arch1-2-x86_64.pkg.tar.zst
TARGET_KVER=6.13.4-arch1-1

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: run with sudo." >&2
  exit 1
fi

echo "=== 0. PRE-FLIGHT ==========================================="
fail=0
for f in "$KERNEL_PKG" "$REVERT_PKG"; do
  if [[ -f "$f" ]]; then echo "  OK   present: $(basename "$f")"
  else echo "  FAIL missing: $f"; fail=1; fi
done

NV_MOD="/usr/lib/modules/${TARGET_KVER}/extramodules/nvidia.ko.zst"
if [[ -f "$NV_MOD" ]]; then
  echo "  OK   nvidia module for ${TARGET_KVER}: $(modinfo "$NV_MOD" 2>/dev/null | awk '/^version:/{print $2}')"
else
  echo "  FAIL no nvidia module at $NV_MOD -- do NOT proceed, the downgrade would"
  echo "       leave you with no driver AND an old kernel."
  fail=1
fi

# WiFi must not depend on DKMS, or a downgrade could cut the network.
WIFI_DRV="$(basename "$(readlink -f /sys/class/net/wlp4s0f3u3/device/driver 2>/dev/null || echo none)")"
echo "  info wlp4s0f3u3 driver: ${WIFI_DRV}"
if [[ "$WIFI_DRV" == "rtl8192cu" ]]; then
  echo "  OK   in-tree driver, present in every kernel -- no DKMS rebuild needed"
elif [[ -z "$(ls -A /var/lib/dkms 2>/dev/null)" ]]; then
  echo "  OK   /var/lib/dkms is empty, so no DKMS module backs the network"
else
  echo "  WARN network may depend on a DKMS module; a 6.13.4 build must exist:"
  ls -A /var/lib/dkms || true
  echo "       if unsure, STOP and plug in ethernet (enp2s0) before continuing."
fi

[[ $fail -eq 0 ]] || { echo "pre-flight FAILED; nothing changed."; exit 1; }
echo

echo "=== 1. INSTALL THE CACHED 6.13.4 KERNEL ====================="
# --needed is wrong here (it would skip a "downgrade"); let pacman replace it.
pacman -U --noconfirm "$KERNEL_PKG"
echo

echo "=== 2. REBUILD INITRAMFS ===================================="
mkinitcpio -P
echo

echo "=== 3. REGENERATE GRUB ======================================"
grub-mkconfig -o /boot/grub/grub.cfg
echo

echo "=== 4. PIN THE KERNEL so this cannot recur =================="
# nvidia et al. are already pinned; `linux` was not, which is exactly how the
# kernel moved out from under the prebuilt module.
if grep -qE '^IgnorePkg.*(^|[[:space:]])linux([[:space:]]|$)' /etc/pacman.conf; then
  echo "  linux already in IgnorePkg"
else
  cp -a /etc/pacman.conf "/etc/pacman.conf.bak.$(date +%Y%m%d%H%M%S)"
  sed -i -E 's/^(IgnorePkg[[:space:]]*=.*)$/\1 linux linux-headers/' /etc/pacman.conf
  echo "  added 'linux linux-headers' to IgnorePkg (backup saved)"
fi
grep -E '^IgnorePkg' /etc/pacman.conf
echo

echo "=== DONE. REBOOT, then verify with: ========================="
cat <<'VERIFY'
  uname -r                      # expect 6.13.4-arch1-1
  nvidia-smi                    # expect the GTX 1660 Ti and a driver version
  nvcc --version                # expect CUDA 12.8
  ip -br addr show wlp4s0f3u3   # expect UP with an address
  python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

REVERT (needs no network -- the package is cached locally):
  sudo pacman -U /var/cache/pacman/pkg/linux-7.1.9.arch1-2-x86_64.pkg.tar.zst
  sudo mkinitcpio -P && sudo grub-mkconfig -o /boot/grub/grub.cfg && sudo reboot
VERIFY
