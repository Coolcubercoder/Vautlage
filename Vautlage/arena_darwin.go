//go:build darwin && (amd64 || arm64)

package vaultage

import (
	"syscall"
	"unsafe"
)

// Darwin VM flags. On macOS an anonymous mmap carries VM allocation flags in
// the `fd` argument rather than in `flags`, which is why the superpage request
// below is passed in the fd position.
const (
	// VM_FLAGS_SUPERPAGE_SIZE_2MB requests a 2MiB superpage mapping.
	//
	// This is accepted only on hosts where the VM subsystem has superpages
	// available; on Apple Silicon it reliably returns EINVAL, and the caller
	// falls back to align-and-trim. It is still attempted because on an
	// Intel Mac with superpages enabled it is the strictly better mapping.
	sysVMFlagsSuperpage2MB = 2 << 16

	// MADV_HUGEPAGE has no darwin equivalent. The closest primitive is
	// MADV_WILLNEED, which prefaults the range but makes no page-size promise.
	sysMadvWillNeed = 3
)

// sysMapAnon maps `length` bytes of private anonymous memory with a direct
// SYS_MMAP syscall.
//
//	mmap(NULL, length, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0)
func sysMapAnon(length uintptr) (unsafe.Pointer, error) {
	r, _, errno := syscall.Syscall6(
		syscall.SYS_MMAP,
		0,
		length,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANON,
		^uintptr(0), // fd = -1
		0,
	)
	if errno != 0 {
		return nil, errno
	}
	return unsafe.Pointer(r), nil
}

// sysMapHuge attempts a 2MiB superpage mapping. Expected to fail on Apple
// Silicon; the arena then falls back to over-allocate-and-trim, which still
// yields a 2MiB-aligned base and therefore still minimises TLB pressure against
// the host's native 16KiB pages.
func sysMapHuge(length uintptr) (unsafe.Pointer, error) {
	if length%HugePageSize != 0 {
		return nil, ErrConfig
	}
	r, _, errno := syscall.Syscall6(
		syscall.SYS_MMAP,
		0,
		length,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANON,
		uintptr(sysVMFlagsSuperpage2MB), // darwin: VM flags ride in the fd slot
		0,
	)
	if errno != 0 {
		return nil, errno
	}
	return unsafe.Pointer(r), nil
}

// sysUnmap releases a mapping with a direct SYS_MUNMAP syscall.
func sysUnmap(p unsafe.Pointer, length uintptr) error {
	if p == nil || length == 0 {
		return nil
	}
	_, _, errno := syscall.Syscall(syscall.SYS_MUNMAP, uintptr(p), length, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

// sysAdviseHuge has no true darwin analogue. MADV_WILLNEED is issued so the
// range is faulted in eagerly; the function reports failure so the arena does
// not claim a huge-page guarantee it cannot make on this platform.
func sysAdviseHuge(p unsafe.Pointer, length uintptr) error {
	_, _, errno := syscall.Syscall(syscall.SYS_MADVISE, uintptr(p), length, sysMadvWillNeed)
	if errno != 0 {
		return errno
	}
	return ErrUnsupportedPlatform
}

// sysPin locks the range into physical RAM with mlock(2).
func sysPin(p unsafe.Pointer, length uintptr) error {
	return syscall.Mlock(unsafe.Slice((*byte)(p), length))
}

// sysUnpin releases the residency guarantee.
func sysUnpin(p unsafe.Pointer, length uintptr) error {
	return syscall.Munlock(unsafe.Slice((*byte)(p), length))
}

// sysPinThreadToCore is a no-op on darwin. XNU exposes no thread-to-CPU binding
// to userspace; THREAD_AFFINITY_POLICY only expresses an affinity *tag* that
// hints two threads should share an L2, and cannot name a specific core.
// Reporting success here would be a lie, so it reports unsupported and the
// engine simply runs without affinity.
func sysPinThreadToCore(core int) error {
	_ = core
	return ErrUnsupportedPlatform
}
