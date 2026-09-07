// Package vaultage implements the extreme-throughput, memory-bounded core engine
// of the Vaultage distributed state fabric.
//
// Vaultage exists to remove the Go runtime heap from the hot path of volatile
// state mutation tracking. On high-density servers the failure mode is physical,
// not algorithmic:
//
//   - Every heap allocation for a transient state record dirties a fresh cache
//     line, evicting live working-set lines from L1/L2.
//   - Millions of small allocations scatter the working set across thousands of
//     4KiB (or 16KiB) pages, blowing out a TLB that holds only ~1500 entries.
//     Every subsequent access becomes a page walk.
//   - The garbage collector must trace those objects, stalling every core with
//     write barriers and assist work at exactly the moment throughput matters.
//   - Kernel demand-paging and swap can silently convert a 200ns access into a
//     multi-millisecond disk fault.
//
// The engine answers each of those with a hardware-level countermeasure:
//
//   - The entire storage matrix is mapped off-heap with a direct SYS_MMAP
//     syscall, so it is invisible to the collector and never traced (arena.go).
//   - The mapping base is forced onto a 2MiB huge-page boundary by
//     over-allocating and trimming, and the kernel is asked to back it with
//     huge pages, collapsing thousands of TLB entries into a handful.
//   - Every resident page is pinned with mlock(2), making the resident set
//     deterministic and eliminating swap-induced tail latency.
//   - Records live in fixed-stride cells whose every 64-bit field sits on an
//     exact 8-byte boundary, asserted at compile time (layout.go), so no atomic
//     ever pays a split-line lock penalty.
//   - Hot cursors are isolated one-per-cache-line (cacheline.go) and workers are
//     sharded per core (fabric.go), so throughput scales with core count instead
//     of collapsing into coherence traffic.
//   - The write and read paths are lock-free CAS turnstiles (ring.go), gated by
//     an atomic high/low-water backpressure valve (gate.go).
//
// The package depends on nothing outside the Go standard library.
package vaultage
