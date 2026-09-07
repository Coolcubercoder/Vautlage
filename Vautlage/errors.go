package vaultage

// Error is a constant-friendly error type. It is a string rather than a struct
// so that every error value in this package is a compile-time constant and
// costs zero allocations to return from the hot path.
type Error string

func (e Error) Error() string { return string(e) }

const (
	// ErrBackpressure is returned when the atomic backpressure gate is closed.
	// The engine crossed its high-water mark and will refuse all ingestion until
	// the consumers drain the matrix below the low-water mark.
	ErrBackpressure = Error("vaultage: ingestion frozen by backpressure gate")

	// ErrShardFull is returned when the target shard's ring has no free slot.
	ErrShardFull = Error("vaultage: shard ring saturated")

	// ErrPayloadTooLarge is returned when a payload exceeds the configured
	// per-cell payload capacity. Cells are fixed-stride by design; the engine
	// never grows, reallocates, or spills to the heap.
	ErrPayloadTooLarge = Error("vaultage: payload exceeds cell payload capacity")

	// ErrClosed is returned once Close has been called.
	ErrClosed = Error("vaultage: engine closed")

	// ErrNotFound is returned by Lookup when a key has no live entry.
	ErrNotFound = Error("vaultage: key not present in index")

	// ErrIndexFull is returned when the open-addressed index has no free bucket
	// within the probe bound.
	ErrIndexFull = Error("vaultage: index probe sequence exhausted")

	// ErrUnsupportedPlatform is returned by the arena allocator on platforms
	// with no raw mmap implementation compiled in.
	ErrUnsupportedPlatform = Error("vaultage: off-heap arena unsupported on this platform")

	// ErrPinFailed is returned when mlock(2) fails and Config.RequirePinned is
	// set, meaning the caller demanded a guaranteed-resident arena.
	ErrPinFailed = Error("vaultage: mlock failed, arena is not pinned to physical RAM")

	// ErrConfig is returned for a structurally invalid configuration.
	ErrConfig = Error("vaultage: invalid configuration")
)
