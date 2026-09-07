package vaultage

import "unsafe"

const (
	// CacheLineSize is the coherence granule of every 64-bit CPU this engine
	// targets: x86-64 (64B lines), and arm64 (64B lines; Apple Silicon
	// prefetches in 128B pairs, which is why hot cursors are given a full
	// 128B stride below rather than a bare 64B one).
	CacheLineSize = 64

	// CoherenceStride is the distance the engine actually puts between two
	// independently-mutated hot variables. It is two cache lines, not one.
	//
	// A single 64B line is enough to defeat classical false sharing, but the
	// adjacent-line prefetchers on both x86-64 (spatial prefetcher, 128B
	// sector pairs) and Apple Silicon speculatively pull the sibling line into
	// the same L2 sector. Two writers landing on sibling lines therefore still
	// ping-pong. A 128B stride puts every hot cursor in its own sector pair and
	// removes the last source of inter-core coherence traffic.
	CoherenceStride = 128

	// WordSize is the width of every atomically-manipulated field in the engine.
	// Nothing narrower is ever placed under a CAS.
	WordSize = 8
)

// padU64 is a single 64-bit atomic cursor that owns an entire coherence stride.
//
// The trailing byte array is not decoration: without it, head and tail would
// share a line, and every producer CAS on tail would invalidate the consumers'
// cached copy of head, converting a lock-free algorithm into a hardware-level
// mutex whose throughput falls as cores are added.
type padU64 struct {
	v uint64
	_ [CoherenceStride - WordSize]byte
}

// load reads the cursor with sequentially-consistent ordering.
func (p *padU64) load() uint64 { return atomicLoad64(&p.v) }

// store publishes the cursor with sequentially-consistent ordering.
func (p *padU64) store(x uint64) { atomicStore64(&p.v, x) }

// add atomically advances the cursor and returns the new value.
func (p *padU64) add(delta uint64) uint64 { return atomicAdd64(&p.v, delta) }

// cas is the turnstile primitive: exactly one racing core observes true.
func (p *padU64) cas(old, new uint64) bool { return atomicCAS64(&p.v, old, new) }

// ---------------------------------------------------------------------------
// Compile-time structural assertions.
//
// Idiom: converting a negative *constant* to uint is a compile error. Asserting
// both (a-b) and (b-a) therefore asserts exact equality at build time, with no
// runtime cost and no test required to catch a layout regression.
// ---------------------------------------------------------------------------

const (
	// padU64 must occupy exactly one coherence stride, no more and no less.
	_ = uint(unsafe.Sizeof(padU64{}) - CoherenceStride)
	_ = uint(CoherenceStride - unsafe.Sizeof(padU64{}))

	// Its atomic word must sit at offset 0 so the CAS address equals the struct
	// address, and must be 8-byte aligned so the lock prefix never straddles a
	// line boundary (a split-line lock is a ~100x latency cliff on x86-64 and a
	// hard SIGBUS on some arm64 configurations).
	_ = uint(unsafe.Offsetof(padU64{}.v) - 0)
	_ = uint(0 - unsafe.Offsetof(padU64{}.v))
	_ = uint(unsafe.Alignof(padU64{}.v) - WordSize)
	_ = uint(WordSize - unsafe.Alignof(padU64{}.v))

	// The engine is 64-bit only. A 32-bit build would give uint64 a 4-byte
	// alignment and silently break every atomic in the package.
	_ = uint(unsafe.Sizeof(uintptr(0)) - 8)
	_ = uint(8 - unsafe.Sizeof(uintptr(0)))
)
