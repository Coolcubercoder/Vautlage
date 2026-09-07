package vaultage

// Record is a view onto one committed state mutation.
//
// It is a value type with no pointer fields other than Payload, and consumers
// are handed a *Record that the worker reuses across iterations, so delivering
// a record costs zero allocations no matter how many billions flow through.
type Record struct {
	// Ticket is the globally ordered sequence number assigned at slot checkout.
	// Tickets are dense and monotonic across the whole engine.
	Ticket uint64
	// Timestamp is the commit time in Unix nanoseconds.
	Timestamp int64
	// Key is the caller-supplied state key.
	Key uint64
	// Epoch is the arena generation. A record whose epoch does not match the
	// live arena is stale and must be discarded.
	Epoch uint64
	// Slot is the physical cell index within the shard's matrix.
	Slot uint64
	// Flags carries FlagCommitted / FlagTombstone / FlagTruncated.
	Flags uint32
	// Checksum is the FNV-1a digest of Payload, or zero when checksums are off.
	Checksum uint32

	// Payload aliases off-heap arena memory directly. NOTHING IS COPIED.
	//
	// These bytes are valid only for the duration of the consumer callback. The
	// instant the callback returns, the slot is released and a producer on
	// another core may overwrite it. Callers that need the bytes afterwards must
	// call CopyPayload.
	Payload []byte

	// Shard is the shard that held this record.
	Shard int
}

// CopyPayload returns a heap-allocated copy of the payload, safe to retain past
// the consumer callback. This is the only allocating operation in the read path,
// and it exists precisely so that the default path can avoid it.
func (r *Record) CopyPayload() []byte {
	if len(r.Payload) == 0 {
		return nil
	}
	out := make([]byte, len(r.Payload))
	copy(out, r.Payload)
	return out
}

// IsTombstone reports whether this record marks a logical deletion.
func (r *Record) IsTombstone() bool { return r.Flags&FlagTombstone != 0 }

// Verify recomputes the payload digest and compares it against the stored one.
// It returns true when checksums are disabled, since there is nothing to refute.
func (r *Record) Verify() bool {
	if r.Checksum == 0 {
		return true
	}
	return checksum32(r.Payload) == r.Checksum
}

// FNV-1a 32-bit. Implemented here rather than imported from hash/fnv because
// the standard implementation allocates a hash.Hash32 interface value per use,
// and this runs on the write path of every record.
const (
	fnvOffset32 = 2166136261
	fnvPrime32  = 16777619
)

// checksum32 digests a payload with FNV-1a.
//
// The zero value is reserved to mean "no checksum", so a payload that genuinely
// digests to zero is nudged to 1. Losing one value out of 2^32 is a better
// trade than an ambiguous sentinel.
//
//go:nosplit
func checksum32(b []byte) uint32 {
	h := uint32(fnvOffset32)
	for i := 0; i < len(b); i++ {
		h ^= uint32(b[i])
		h *= fnvPrime32
	}
	if h == 0 {
		return 1
	}
	return h
}
