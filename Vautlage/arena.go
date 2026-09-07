package vaultage

import (
	"syscall"
	"unsafe"
)

// HugePageSize is the huge-page granule the arena aligns to: 2MiB.
//
// This is the x86-64 PMD-level page size and the arm64 contiguous-PTE/block
// size. Aligning here is what converts 512 individual 4KiB TLB entries into a
// single entry: a 1GiB matrix addressed with 4KiB pages needs 262,144 TLB
// entries against a hardware TLB holding roughly 1,500-3,000, so the working
// set is guaranteed to thrash. The same matrix on 2MiB pages needs 512 entries
// and fits.
const HugePageSize uintptr = 2 << 20

// Arena is a contiguous span of anonymous memory obtained straight from the
// kernel, outside the Go heap.
//
// Because the span is not Go-heap memory:
//   - the garbage collector never scans it, so millions of live records cost
//     exactly zero mark work and zero write barriers;
//   - it never moves, so raw addresses derived from it stay valid for the
//     lifetime of the mapping;
//   - it must contain no Go pointers, only plain bytes and integers. Every
//     structure the engine writes into an arena obeys that rule by construction.
//
// An Arena is created once at start-up and released once at shutdown. There is
// no allocator on the hot path, because there is no allocation on the hot path.
type Arena struct {
	// base is the huge-page-aligned first usable byte.
	base unsafe.Pointer
	// size is the usable length in bytes, always a multiple of HugePageSize.
	size uintptr

	// raw/rawSize describe the mapping actually handed back by mmap, which may
	// be larger than [base, base+size) when alignment had to be forced by
	// over-allocating. munmap must be given the mapping, not the aligned view.
	raw     unsafe.Pointer
	rawSize uintptr

	pageSize uintptr

	// hugeNative is true when the kernel satisfied the mapping from the
	// explicit huge-page pool (Linux MAP_HUGETLB) rather than from ordinary
	// anonymous memory promoted by khugepaged.
	hugeNative bool
	// hugeAdvised is true when madvise(MADV_HUGEPAGE) was accepted.
	hugeAdvised bool
	// pinned is true when mlock(2) succeeded and the span is guaranteed
	// resident in physical RAM.
	pinned bool
}

// ArenaStats is a snapshot of an arena's physical properties.
type ArenaStats struct {
	Base            uintptr // virtual address of the first usable byte
	Size            uintptr // usable bytes
	MappedSize      uintptr // bytes actually returned by mmap
	PageSize        uintptr // host hardware page size
	HugePageAligned bool    // base sits on a 2MiB boundary
	HugePagesNative bool    // served from the explicit huge-page pool
	HugePagesAdvise bool    // madvise(MADV_HUGEPAGE) accepted
	Pinned          bool    // mlock(2) succeeded; span cannot be swapped
}

// NewArena maps at least `request` bytes of anonymous memory off-heap, forces
// the usable base onto a HugePageSize boundary, asks the kernel to back the span
// with huge pages, prefaults every page, and pins the span into physical RAM.
//
// The alignment strategy has two tiers, because only one of them is available on
// any given host:
//
//	Tier 1 (Linux, explicit huge pages). Ask for MAP_HUGETLB directly. The
//	kernel serves the mapping from the reserved huge-page pool, and such a
//	mapping is 2MiB aligned by definition. This is the only tier that
//	*guarantees* huge pages rather than merely making them likely. It requires
//	the operator to have reserved pages (vm.nr_hugepages) and fails otherwise.
//
//	Tier 2 (everywhere, over-allocate and trim). mmap request+HugePageSize
//	bytes, round the returned address up to the next 2MiB boundary, then munmap
//	the slack before and after the aligned window. The kernel never promises an
//	aligned address on its own (this host returned 0x10bcb0000, which is not
//	2MiB aligned), so trimming is the only portable way to force alignment. The
//	slack is returned to the OS immediately, so the process pays no RSS for it.
//	Transparent huge pages are then requested with madvise(MADV_HUGEPAGE).
//
// If RequirePinned is set and mlock(2) fails, the arena is torn down and
// ErrPinFailed is returned rather than silently running a swappable arena.
func NewArena(request uintptr, requireHuge, requirePinned bool) (*Arena, error) {
	if request == 0 {
		return nil, ErrConfig
	}

	pageSize := uintptr(syscall.Getpagesize())
	if !isPow2(pageSize) {
		return nil, ErrConfig
	}

	// The usable span is always a whole number of huge pages. Rounding here,
	// rather than at the call site, is what lets every downstream offset
	// computation assume a huge-page-sized, huge-page-aligned window.
	size := alignUp(request, HugePageSize)

	a := &Arena{size: size, pageSize: pageSize}

	// ---- Tier 1: explicit huge pages -------------------------------------
	if p, err := sysMapHuge(size); err == nil {
		a.raw, a.rawSize = p, size
		a.base = p
		a.hugeNative = true
	} else {
		if requireHuge {
			return nil, err
		}

		// ---- Tier 2: over-allocate and trim ------------------------------
		//
		//   raw                aligned base            aligned end        raw end
		//    |<-- head slack -->|<------- size ------->|<-- tail slack -->|
		//    |                  |                      |                  |
		//    +------------------+----------------------+------------------+
		//
		// Over-allocating by exactly one huge page is sufficient: for any
		// returned address, the next 2MiB boundary is less than HugePageSize
		// bytes away, so the aligned window always fits inside the mapping.
		rawSize := size + HugePageSize
		p, err := sysMapAnon(rawSize)
		if err != nil {
			return nil, err
		}

		rawAddr := uintptr(p)
		baseAddr := alignUp(rawAddr, HugePageSize)

		// Physical address arithmetic: the aligned base is `headSlack` bytes
		// into the mapping, and the tail slack is whatever remains after the
		// usable window.
		headSlack := baseAddr - rawAddr
		tailSlack := rawSize - headSlack - size

		if headSlack > 0 {
			// Release [raw, alignedBase). Ignoring the error is not an option:
			// a failed trim would leak an unreferenced mapping.
			if err := sysUnmap(p, headSlack); err != nil {
				_ = sysUnmap(p, rawSize)
				return nil, err
			}
		}
		if tailSlack > 0 {
			tailPtr := unsafe.Add(p, headSlack+size)
			if err := sysUnmap(tailPtr, tailSlack); err != nil {
				_ = sysUnmap(unsafe.Add(p, headSlack), size)
				return nil, err
			}
		}

		a.base = unsafe.Add(p, headSlack)
		a.raw, a.rawSize = a.base, size

		// Ask for transparent huge pages over the aligned window. Because the
		// window is both 2MiB aligned and a whole number of 2MiB pages,
		// khugepaged can collapse it without splitting.
		if err := sysAdviseHuge(a.base, size); err == nil {
			a.hugeAdvised = true
		} else if requireHuge {
			_ = sysUnmap(a.base, size)
			return nil, err
		}
	}

	// Alignment is a hard invariant for everything downstream: the ring's cell
	// arithmetic assumes it when it claims every atomic is 8-byte aligned.
	if !isAligned(uintptr(a.base), HugePageSize) {
		_ = sysUnmap(a.raw, a.rawSize)
		return nil, ErrConfig
	}

	// Prefault every page before the workload starts. mlock will populate too,
	// but doing it explicitly means the arena is fully resident even when
	// pinning is unavailable, so no producer ever eats a minor page fault on
	// the hot path.
	a.prefault()

	// Pin the span. Without this the kernel is free to page the matrix out
	// under memory pressure, converting a ~100ns cache miss into a multi-
	// millisecond disk fault and destroying tail-latency determinism.
	if err := sysPin(a.base, a.size); err == nil {
		a.pinned = true
	} else if requirePinned {
		_ = sysUnmap(a.raw, a.rawSize)
		return nil, ErrPinFailed
	}

	return a, nil
}

// prefault walks the span one hardware page at a time, forcing the kernel to
// allocate and zero every physical frame up front.
//
// The write must be a real store: a read of anonymous memory can be satisfied by
// the shared zero page, which allocates no frame and leaves a copy-on-write
// fault waiting on the hot path. Writing zero to a byte that is already zero
// still dirties the page and forces a private frame.
func (a *Arena) prefault() {
	stride := a.pageSize
	if a.hugeNative || a.hugeAdvised {
		// On a huge-page-backed span one touch per 2MiB is enough, and touching
		// per 4KiB would risk splitting the huge page into base pages.
		stride = HugePageSize
	}
	for off := uintptr(0); off < a.size; off += stride {
		p := (*byte)(unsafe.Add(a.base, off))
		*p = 0
	}
}

// Base returns the huge-page-aligned first usable address of the span.
//
//go:nosplit
func (a *Arena) Base() unsafe.Pointer { return a.base }

// Size returns the usable length of the span in bytes.
//
//go:nosplit
func (a *Arena) Size() uintptr { return a.size }

// At returns the address `off` bytes into the arena.
//
//	addr = base + off
//
// This is the fundamental physical address computation of the whole engine:
// every cell, header field, payload and index bucket is reached by some
// composition of this one operation.
//
//go:nosplit
func (a *Arena) At(off uintptr) unsafe.Pointer { return unsafe.Add(a.base, off) }

// Bytes exposes the whole span as a byte slice aliasing the mapping. No copy is
// made. Intended for zeroing, checksumming, and snapshot I/O, not for the hot
// path.
func (a *Arena) Bytes() []byte {
	return unsafe.Slice((*byte)(a.base), a.size)
}

// Zero clears the span. Used when an arena generation is retired and reissued.
func (a *Arena) Zero() {
	b := a.Bytes()
	for i := range b {
		b[i] = 0
	}
}

// Stats returns a snapshot of the arena's physical properties.
func (a *Arena) Stats() ArenaStats {
	return ArenaStats{
		Base:            uintptr(a.base),
		Size:            a.size,
		MappedSize:      a.rawSize,
		PageSize:        a.pageSize,
		HugePageAligned: isAligned(uintptr(a.base), HugePageSize),
		HugePagesNative: a.hugeNative,
		HugePagesAdvise: a.hugeAdvised,
		Pinned:          a.pinned,
	}
}

// Close unpins and unmaps the span. Every address previously derived from this
// arena becomes invalid the instant this returns; dereferencing one afterwards
// is a segmentation fault, not a Go panic. The engine guarantees all workers
// have been joined before this is called.
func (a *Arena) Close() error {
	if a.base == nil {
		return nil
	}
	if a.pinned {
		_ = sysUnpin(a.base, a.size)
		a.pinned = false
	}
	err := sysUnmap(a.raw, a.rawSize)
	a.base, a.raw, a.size, a.rawSize = nil, nil, 0, 0
	return err
}
