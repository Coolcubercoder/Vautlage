package vaultage

import "unsafe"

// ---------------------------------------------------------------------------
// CELL LAYOUT
//
// The off-heap span is sliced into fixed-stride cells. Each cell is:
//
//   byte 0                     byte 64                     byte 64+payloadCap
//   +--------------------------+---------------------------+
//   |      cellHeader (64B)    |        payload bytes      |  ... pad to 64B ...
//   +--------------------------+---------------------------+
//   |<--- exactly one cache line --->|
//
// The header is exactly one cache line wide, so touching a record's metadata
// costs precisely one line fill and never straddles two lines. The payload
// begins on the next line, so a producer streaming a payload does not
// repeatedly dirty the header line it has already published.
//
// Header field map (every field on an exact 8-byte boundary):
//
//   offset  width  field      purpose
//   ------  -----  ---------  ---------------------------------------------
//        0      8  seq        turnstile sequence: the lock-free slot state
//        8      8  ticket     globally ordered ticket id assigned at checkout
//       16      8  timestamp  commit time, unix nanoseconds
//       24      8  key        caller state key
//       32      8  length     payload length in bytes
//       40      8  epoch      arena generation, invalidates stale index entries
//       48      4  checksum   payload integrity digest
//       52      4  flags      record flags
//       56      8  reserved   pad to a full line; keeps stride 8-byte exact
//
// `seq` is deliberately at offset 0. The turnstile CAS is the single hottest
// instruction in the engine, and placing it at offset 0 makes the atomic's
// address identical to the cell's base address: one address computation, no
// displacement arithmetic in the inner loop.
// ---------------------------------------------------------------------------

// HeaderSize is the fixed per-cell metadata size: exactly one cache line.
const HeaderSize = 64

// Header field offsets. These are the *only* values the hot path uses; the
// cellHeader struct below exists solely to pin these numbers at compile time.
const (
	offSeq       = 0
	offTicket    = 8
	offTimestamp = 16
	offKey       = 24
	offLength    = 32
	offEpoch     = 40
	offChecksum  = 48
	offFlags     = 52
	offReserved  = 56
)

// Record flag bits stored in the header's flags word.
const (
	// FlagCommitted marks a cell whose payload write has fully landed.
	FlagCommitted uint32 = 1 << 0
	// FlagTombstone marks a logical deletion of the key.
	FlagTombstone uint32 = 1 << 1
	// FlagTruncated marks a record whose payload was clipped to cell capacity.
	FlagTruncated uint32 = 1 << 2
)

// cellHeader is a layout witness. The engine never allocates, dereferences, or
// casts to this type at runtime; it exists purely so the compiler can prove the
// offset constants above match a real, layout-stable Go structure. If a field is
// ever reordered, resized, or repadded, the const block below fails to build.
type cellHeader struct {
	seq       uint64
	ticket    uint64
	timestamp uint64
	key       uint64
	length    uint64
	epoch     uint64
	checksum  uint32
	flags     uint32
	reserved  uint64
}

// Compile-time offset assertions. Each pair asserts exact equality: converting a
// negative constant to uint is a build error, so asserting both directions of
// the difference pins the offset to a single legal value.
const (
	_ = uint(unsafe.Offsetof(cellHeader{}.seq) - offSeq)
	_ = uint(offSeq - unsafe.Offsetof(cellHeader{}.seq))

	_ = uint(unsafe.Offsetof(cellHeader{}.ticket) - offTicket)
	_ = uint(offTicket - unsafe.Offsetof(cellHeader{}.ticket))

	_ = uint(unsafe.Offsetof(cellHeader{}.timestamp) - offTimestamp)
	_ = uint(offTimestamp - unsafe.Offsetof(cellHeader{}.timestamp))

	_ = uint(unsafe.Offsetof(cellHeader{}.key) - offKey)
	_ = uint(offKey - unsafe.Offsetof(cellHeader{}.key))

	_ = uint(unsafe.Offsetof(cellHeader{}.length) - offLength)
	_ = uint(offLength - unsafe.Offsetof(cellHeader{}.length))

	_ = uint(unsafe.Offsetof(cellHeader{}.epoch) - offEpoch)
	_ = uint(offEpoch - unsafe.Offsetof(cellHeader{}.epoch))

	_ = uint(unsafe.Offsetof(cellHeader{}.checksum) - offChecksum)
	_ = uint(offChecksum - unsafe.Offsetof(cellHeader{}.checksum))

	_ = uint(unsafe.Offsetof(cellHeader{}.flags) - offFlags)
	_ = uint(offFlags - unsafe.Offsetof(cellHeader{}.flags))

	_ = uint(unsafe.Offsetof(cellHeader{}.reserved) - offReserved)
	_ = uint(offReserved - unsafe.Offsetof(cellHeader{}.reserved))

	// The header is exactly one cache line: no trailing compiler padding, no
	// accidental spill into the payload line.
	_ = uint(unsafe.Sizeof(cellHeader{}) - HeaderSize)
	_ = uint(HeaderSize - unsafe.Sizeof(cellHeader{}))

	// Every 64-bit field carries 8-byte alignment. This is what makes the
	// LOCK CMPXCHG / LDAXR-STLXR on these addresses single-line and fault-free.
	_ = uint(unsafe.Alignof(cellHeader{}.seq) - WordSize)
	_ = uint(WordSize - unsafe.Alignof(cellHeader{}.seq))
	_ = uint(unsafe.Alignof(cellHeader{}.ticket) - WordSize)
	_ = uint(WordSize - unsafe.Alignof(cellHeader{}.ticket))
	_ = uint(unsafe.Alignof(cellHeader{}.timestamp) - WordSize)
	_ = uint(WordSize - unsafe.Alignof(cellHeader{}.timestamp))
	_ = uint(unsafe.Alignof(cellHeader{}.key) - WordSize)
	_ = uint(WordSize - unsafe.Alignof(cellHeader{}.key))
	_ = uint(unsafe.Alignof(cellHeader{}.length) - WordSize)
	_ = uint(WordSize - unsafe.Alignof(cellHeader{}.length))
	_ = uint(unsafe.Alignof(cellHeader{}.epoch) - WordSize)
	_ = uint(WordSize - unsafe.Alignof(cellHeader{}.epoch))

	// Every 64-bit offset is itself a multiple of 8. Asserted independently of
	// the equality checks above so that renumbering the offsets cannot silently
	// introduce a straddling field.
	_ = uint(0 - offSeq%WordSize)
	_ = uint(0 - offTicket%WordSize)
	_ = uint(0 - offTimestamp%WordSize)
	_ = uint(0 - offKey%WordSize)
	_ = uint(0 - offLength%WordSize)
	_ = uint(0 - offEpoch%WordSize)
	_ = uint(0 - offReserved%WordSize)

	// The 32-bit pair shares one 8-byte word: checksum at +0, flags at +4.
	_ = uint(0 - offChecksum%WordSize)
	_ = uint(offFlags - offChecksum - 4)
	_ = uint(4 - (offFlags - offChecksum))
)

// ---------------------------------------------------------------------------
// RAW ADDRESS ARITHMETIC
//
// Every accessor below is the explicit physical address computation used to
// reach a field inside the flat off-heap block. There is no slice indexing, no
// bounds check, and no intermediate Go object: the compiler emits a shift, an
// add, and the memory instruction.
// ---------------------------------------------------------------------------

// cellAt returns the base address of cell `idx` in a matrix whose first cell
// begins at `base` and whose cells are `stride` bytes apart.
//
//	addr = base + idx*stride
//
// `stride` is a multiple of CacheLineSize and `base` is 2MiB aligned, so the
// result is always 64-byte aligned and therefore every header field within the
// cell lands on an exact 8-byte boundary.
//
//go:nosplit
func cellAt(base unsafe.Pointer, idx, stride uintptr) unsafe.Pointer {
	return unsafe.Add(base, idx*stride)
}

// fieldAt returns the address of a header field within a cell.
//
//	addr = cellBase + fieldOffset
//
//go:nosplit
func fieldAt(cellBase unsafe.Pointer, off uintptr) unsafe.Pointer {
	return unsafe.Add(cellBase, off)
}

// payloadAt returns the address of a cell's payload region, which begins on the
// cache line immediately after the 64-byte header.
//
//	addr = cellBase + HeaderSize
//
//go:nosplit
func payloadAt(cellBase unsafe.Pointer) unsafe.Pointer {
	return unsafe.Add(cellBase, HeaderSize)
}

// payloadSlice materialises a Go byte slice that aliases the off-heap payload
// region directly. No copy is made and no heap memory is allocated: the slice
// header simply points into mmap'd pages.
//
// The returned slice is valid only while the caller owns the slot. Once the slot
// is released back to the ring a producer may overwrite these bytes at any
// instant, so callers needing durable bytes must copy them out.
//
//go:nosplit
func payloadSlice(cellBase unsafe.Pointer, n uintptr) []byte {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(payloadAt(cellBase)), n)
}

// copyIntoPayload writes src into the cell's payload region with a single
// runtime memmove and returns the number of bytes written. The destination is
// off-heap, so this copy is invisible to the garbage collector and emits no
// write barrier.
//
//go:nosplit
func copyIntoPayload(cellBase unsafe.Pointer, src []byte) uintptr {
	if len(src) == 0 {
		return 0
	}
	dst := unsafe.Slice((*byte)(payloadAt(cellBase)), len(src))
	return uintptr(copy(dst, src))
}

// alignUp rounds n up to the next multiple of align, which must be a power of
// two. Branchless form: add align-1, then mask the low bits away.
//
//go:nosplit
func alignUp(n, align uintptr) uintptr {
	return (n + align - 1) &^ (align - 1)
}

// isAligned reports whether addr sits on an `align` boundary.
//
//go:nosplit
func isAligned(addr, align uintptr) bool {
	return addr&(align-1) == 0
}

// isPow2 reports whether n is a non-zero power of two.
func isPow2(n uintptr) bool { return n != 0 && n&(n-1) == 0 }

// cellStrideFor computes the fixed cell stride for a given payload capacity:
// the header plus the payload, rounded up so that every cell begins on a cache
// line. That rounding is what keeps the matrix layout-stable and guarantees
// cell N's header can never share a line with cell N-1's payload tail.
func cellStrideFor(payloadCap uintptr) uintptr {
	return alignUp(HeaderSize+payloadCap, CacheLineSize)
}
