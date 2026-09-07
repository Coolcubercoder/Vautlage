package vaultage

import (
	"sync/atomic"
	"unsafe"
)

// This file is the single choke point through which every atomic in the engine
// passes. It exists for three reasons:
//
//  1. The storage matrix lives off-heap, so its fields are addressed by raw
//     pointer arithmetic rather than as Go struct fields. sync/atomic requires
//     *uint64, so the conversion happens here, once, under a documented
//     alignment contract instead of being scattered across the hot paths.
//
//  2. Address math is carried in unsafe.Pointer and advanced with unsafe.Add,
//     never round-tripped through uintptr. Off-heap addresses are immovable, so
//     a uintptr round trip would in fact be safe here, but keeping the pointer
//     form makes the code provably correct to `go vet -unsafeptr` and immune to
//     any future change in how the runtime treats derived addresses.
//
//  3. Every helper is a one-line wrapper the compiler inlines into its caller,
//     so the emitted code is the bare LOCK CMPXCHG (x86-64) or LDAXR/STLXR
//     (arm64) sequence with no call frame.
//
// ALIGNMENT CONTRACT: every pointer passed to the helpers below is guaranteed
// 8-byte aligned by construction, because
//   arena base   is 2MiB aligned          (arena.go),
//   cell stride  is a multiple of 64      (layout.go: cellStrideFor), and
//   field offset is a multiple of 8       (layout.go: asserted at compile time).
// An unaligned atomic is therefore structurally unreachable, not merely
// untested. This matters physically: a lock instruction that straddles two
// cache lines degrades from a cache-local operation into a bus-wide split lock,
// costing on the order of 100x and stalling every other core on the socket.

func atomicLoad64(p *uint64) uint64          { return atomic.LoadUint64(p) }
func atomicStore64(p *uint64, v uint64)      { atomic.StoreUint64(p, v) }
func atomicAdd64(p *uint64, d uint64) uint64 { return atomic.AddUint64(p, d) }
func atomicCAS64(p *uint64, old, new uint64) bool {
	return atomic.CompareAndSwapUint64(p, old, new)
}

// ptrLoad64 atomically loads the 64-bit word at an off-heap address.
//
//go:nosplit
func ptrLoad64(p unsafe.Pointer) uint64 {
	return atomic.LoadUint64((*uint64)(p))
}

// ptrStore64 atomically publishes a 64-bit word to an off-heap address.
//
// In the Go memory model this store is sequentially consistent, so every plain
// write issued before it is visible to any core that subsequently observes the
// new value. That is precisely the release edge the ring's commit protocol
// relies on to publish a payload without a lock or a fence intrinsic.
//
//go:nosplit
func ptrStore64(p unsafe.Pointer, v uint64) {
	atomic.StoreUint64((*uint64)(p), v)
}

// ptrCAS64 is the ticket turnstile applied directly to off-heap memory: of all
// cores racing on this word, exactly one observes true.
//
//go:nosplit
func ptrCAS64(p unsafe.Pointer, old, new uint64) bool {
	return atomic.CompareAndSwapUint64((*uint64)(p), old, new)
}

// ptrAdd64 atomically advances a 64-bit counter at an off-heap address.
//
//go:nosplit
func ptrAdd64(p unsafe.Pointer, d uint64) uint64 {
	return atomic.AddUint64((*uint64)(p), d)
}

// ptrLoad32 atomically loads a 32-bit word at an off-heap address.
//
//go:nosplit
func ptrLoad32(p unsafe.Pointer) uint32 {
	return atomic.LoadUint32((*uint32)(p))
}

// ptrStore32 atomically publishes a 32-bit word to an off-heap address.
//
//go:nosplit
func ptrStore32(p unsafe.Pointer, v uint32) {
	atomic.StoreUint32((*uint32)(p), v)
}
