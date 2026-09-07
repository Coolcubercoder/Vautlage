//go:build linux && (amd64 || arm64)

package vaultage

import (
	"syscall"
	"unsafe"
)

// Linux memory-management flags that the standard syscall package does not
// export. The values are stable kernel ABI and identical on amd64 and arm64,
// which is why this file is constrained to those two architectures rather than
// to `linux` generally (32-bit x86, for one, routes mmap through a different
// entry point with a page-shifted offset argument).
const (
	// MAP_HUGETLB serves the mapping from the reserved huge-page pool.
	sysMapHugetlb = 0x40000
	// MAP_HUGE_SHIFT is where the log2 of the requested page size is encoded.
	sysMapHugeShift = 26
	// MAP_HUGE_2MB: log2(2MiB) == 21, shifted into the flag field.
	sysMapHuge2MB = 21 << sysMapHugeShift
	// MADV_HUGEPAGE asks khugepaged to back the range with transparent huge
	// pages and permits synchronous collapse on fault.
	sysMadvHugepage = 14
	// MADV_DONTFORK keeps the arena out of a forked child, so a fork() cannot
	// mark the whole matrix copy-on-write and silently double its RSS.
	sysMadvDontFork = 10
)

// sysMapAnon maps `length` bytes of private anonymous memory with a direct
// SYS_MMAP syscall.
//
//	mmap(NULL, length, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
//
// The syscall is issued raw rather than through syscall.Mmap because the
// standard wrapper maintains its own mapping bookkeeping and returns a Go slice,
// both of which impose exactly the runtime coupling this engine exists to avoid.
func sysMapAnon(length uintptr) (unsafe.Pointer, error) {
	r, _, errno := syscall.Syscall6(
		syscall.SYS_MMAP,
		0,      // addr: let the kernel choose
		length, // length
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS,
		^uintptr(0), // fd = -1
		0,           // offset
	)
	if errno != 0 {
		return nil, errno
	}
	return unsafe.Pointer(r), nil
}

// sysMapHuge attempts to map `length` bytes from the explicit 2MiB huge-page
// pool. A mapping served this way is 2MiB aligned by construction, so no
// trimming is needed.
//
// This fails with ENOMEM when the operator has not reserved pages
// (vm.nr_hugepages) and with EINVAL when length is not a multiple of 2MiB. Both
// are ordinary conditions on a general-purpose host, and the caller falls back
// to the align-and-trim path.
func sysMapHuge(length uintptr) (unsafe.Pointer, error) {
	if length%HugePageSize != 0 {
		return nil, ErrConfig
	}
	r, _, errno := syscall.Syscall6(
		syscall.SYS_MMAP,
		0,
		length,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS|sysMapHugetlb|sysMapHuge2MB,
		^uintptr(0),
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

// sysAdviseHuge requests transparent huge pages over the range, and detaches it
// from fork() so a child process cannot force copy-on-write across the matrix.
func sysAdviseHuge(p unsafe.Pointer, length uintptr) error {
	_, _, errno := syscall.Syscall(syscall.SYS_MADVISE, uintptr(p), length, sysMadvHugepage)
	if errno != 0 {
		return errno
	}
	// Best effort: an old kernel without MADV_DONTFORK must not fail the mapping.
	_, _, _ = syscall.Syscall(syscall.SYS_MADVISE, uintptr(p), length, sysMadvDontFork)
	return nil
}

// sysPin locks the range into physical RAM with mlock(2), guaranteeing that no
// access to the matrix can ever take a major page fault.
func sysPin(p unsafe.Pointer, length uintptr) error {
	return syscall.Mlock(unsafe.Slice((*byte)(p), length))
}

// sysUnpin releases the residency guarantee.
func sysUnpin(p unsafe.Pointer, length uintptr) error {
	return syscall.Munlock(unsafe.Slice((*byte)(p), length))
}

// sysPinThreadToCore binds the calling OS thread to one logical CPU with
// sched_setaffinity(2).
//
// This is the other half of cache isolation. Padding stops two cores from
// sharing a line; affinity stops one worker from migrating between cores and
// dragging its entire working set through a cold L1/L2 on the destination.
// The caller must already hold runtime.LockOSThread.
func sysPinThreadToCore(core int) error {
	// cpu_set_t is a bitmask; 1024 bits is the kernel default.
	const setWords = 1024 / 64
	var set [setWords]uint64
	if core < 0 || core >= 1024 {
		return ErrConfig
	}
	set[core/64] |= 1 << (uint(core) % 64)

	_, _, errno := syscall.Syscall(
		syscall.SYS_SCHED_SETAFFINITY,
		0, // pid 0 == calling thread
		unsafe.Sizeof(set),
		uintptr(unsafe.Pointer(&set[0])),
	)
	if errno != 0 {
		return errno
	}
	return nil
}
