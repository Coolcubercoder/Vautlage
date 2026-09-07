package vaultage

import "unsafe"

// ---------------------------------------------------------------------------
// OFF-HEAP KEY INDEX
//
// A lock-free open-addressed hash table mapping a state key to the physical cell
// holding that key's most recent mutation. Like the ring, it lives entirely in
// arena memory: no Go map, no buckets on the heap, no GC pressure, and no
// rehashing pause. Its capacity is fixed at construction, which is the point --
// a memory-bounded engine cannot have an index that grows without bound.
//
// Bucket layout, 32 bytes, two buckets per cache line:
//
//   offset  width  field   purpose
//   ------  -----  ------  --------------------------------------------------
//        0      8  key     0 == empty; claimed with a single CAS
//        8      8  ticket  ticket of the indexed record
//       16      8  slot    physical cell index within the shard matrix
//       24      8  pad     keeps the bucket a power-of-two size and 8-aligned
//
// CONCURRENCY. Key 0 is reserved as the empty sentinel, so a bucket is claimed
// with one CompareAndSwapUint64 on the key word -- no state machine, no lock,
// no claiming interval during which readers must block. A reader that arrives
// between the key CAS and the value stores sees ticket == 0 and treats the
// bucket as a miss; it never blocks and never reads a half-written value.
//
// AUTHORITY. The index is a hint, not the source of truth. Lookup always
// validates its answer against the cell header it points at: if that cell has
// since been recycled by a producer on another lap, its key no longer matches
// and Lookup reports a miss rather than returning another key's data. This is
// what allows the whole structure to stay lock-free without version counters,
// hazard pointers, or epoch reclamation.
// ---------------------------------------------------------------------------

// Index bucket size and field offsets.
const (
	IndexBucketSize = 32

	offIdxKey    = 0
	offIdxTicket = 8
	offIdxSlot   = 16
	offIdxPad    = 24
)

// indexBucket is a layout witness, exactly as cellHeader is for the matrix. It
// is never instantiated at run time.
type indexBucket struct {
	key    uint64
	ticket uint64
	slot   uint64
	pad    uint64
}

const (
	_ = uint(unsafe.Offsetof(indexBucket{}.key) - offIdxKey)
	_ = uint(offIdxKey - unsafe.Offsetof(indexBucket{}.key))
	_ = uint(unsafe.Offsetof(indexBucket{}.ticket) - offIdxTicket)
	_ = uint(offIdxTicket - unsafe.Offsetof(indexBucket{}.ticket))
	_ = uint(unsafe.Offsetof(indexBucket{}.slot) - offIdxSlot)
	_ = uint(offIdxSlot - unsafe.Offsetof(indexBucket{}.slot))
	_ = uint(unsafe.Offsetof(indexBucket{}.pad) - offIdxPad)
	_ = uint(offIdxPad - unsafe.Offsetof(indexBucket{}.pad))

	_ = uint(unsafe.Sizeof(indexBucket{}) - IndexBucketSize)
	_ = uint(IndexBucketSize - unsafe.Sizeof(indexBucket{}))

	// Buckets must tile a cache line exactly, so no bucket ever straddles two
	// lines and no CAS on a key word can become a split lock.
	_ = uint(0 - CacheLineSize%IndexBucketSize)
	_ = uint(0 - offIdxKey%WordSize)
	_ = uint(0 - offIdxTicket%WordSize)
	_ = uint(0 - offIdxSlot%WordSize)
)

// maxProbe bounds the linear probe sequence.
//
// It is deliberately small and cache-shaped: 16 buckets is 512 bytes, eight
// cache lines, walked linearly so the hardware prefetcher covers it. Beyond
// that distance a table is pathologically loaded and the honest answer is to
// report ErrIndexFull rather than degrade into an unbounded scan on the hot
// path.
const maxProbe = 16

// keyIndex is the off-heap hash table for one shard.
type keyIndex struct {
	base    unsafe.Pointer // bucket 0 inside the arena
	buckets uint64         // power of two
	mask    uint64
	_       [CoherenceStride - 24]byte

	live      padU64 // occupied buckets
	collides  padU64 // probe steps taken beyond the home bucket
	overflows padU64 // insertions abandoned after maxProbe
	reclaims  padU64 // dead buckets recycled in place
}

// indexBytes returns the arena bytes required for `buckets` buckets.
func indexBytes(buckets uint64) uintptr {
	return uintptr(buckets) * IndexBucketSize
}

// initIndex binds the table to arena memory. The caller has already zeroed the
// arena, so every key word starts at the empty sentinel; the loop below is
// explicit anyway, because relying on incidental zeroing is how a reused arena
// generation ends up serving another generation's keys.
func initIndex(x *keyIndex, base unsafe.Pointer, buckets uint64) error {
	if buckets == 0 || !isPow2(uintptr(buckets)) {
		return ErrConfig
	}
	if !isAligned(uintptr(base), CacheLineSize) {
		return ErrConfig
	}
	x.base = base
	x.buckets = buckets
	x.mask = buckets - 1

	for i := uint64(0); i < buckets; i++ {
		b := x.bucket(i)
		ptrStore64(fieldAt(b, offIdxKey), 0)
		ptrStore64(fieldAt(b, offIdxTicket), 0)
		ptrStore64(fieldAt(b, offIdxSlot), 0)
		ptrStore64(fieldAt(b, offIdxPad), 0)
	}
	x.live.store(0)
	x.collides.store(0)
	x.overflows.store(0)
	x.reclaims.store(0)
	return nil
}

// bucket returns the address of bucket i.
//
//	addr = base + i*32
//
//go:nosplit
func (x *keyIndex) bucket(i uint64) unsafe.Pointer {
	return unsafe.Add(x.base, uintptr(i)*IndexBucketSize)
}

// hashKey is the SplitMix64 finalizer.
//
// State keys in real deployments are frequently dense integers, sequential ids,
// or values sharing low bits. Masking such a key straight into a bucket index
// would pile every key onto a handful of home buckets. This finalizer avalanches
// every input bit across all 64 output bits in four instructions with no table
// lookup and no memory traffic.
//
//go:nosplit
func hashKey(k uint64) uint64 {
	k ^= k >> 30
	k *= 0xbf58476d1ce4e5b9
	k ^= k >> 27
	k *= 0x94d049bb133111eb
	k ^= k >> 31
	return k
}

// upsert points `key` at the record identified by (ticket, slot).
//
// The write path is a single CAS in the common case:
//
//   - Home bucket empty: one CompareAndSwapUint64 claims it. Values are stored
//     afterwards; a concurrent reader that catches the interval sees ticket == 0
//     and reports a miss rather than blocking.
//
//   - Home bucket already holds this key: the ticket word is advanced with a
//     monotonic CAS loop, so an older mutation arriving late on another core can
//     never overwrite a newer one. Ticket order is total across the engine, so
//     this comparison is exact rather than heuristic.
//
// When the probe sequence is full, a second pass reclaims dead buckets. A bucket
// is dead when the cell it points at has been recycled by the ring and no longer
// holds its key, which is the steady state for any key that has aged out of the
// matrix. Without this pass the table would fill permanently after `buckets`
// distinct keys and reject every new one thereafter, even though nearly all of
// its entries point at cells that were overwritten long ago.
//
// Key 0 is rejected: it is the empty sentinel.
//
//go:nosplit
func (x *keyIndex) upsert(key, ticket, slot uint64, r *ring) error {
	if key == 0 {
		return ErrConfig
	}
	h := hashKey(key) & x.mask

	// Bounded outer loop rather than recursion: this is a //go:nosplit function
	// on the hottest path in the engine, and a recursive retry would both defeat
	// the nosplit stack check and put an unbounded frame chain under a producer.
	// Two attempts is sufficient -- the second only runs when the table shifted
	// underneath the first, and a third would be chasing a table under such
	// churn that ErrIndexFull is the honest answer.
	for attempt := 0; attempt < 2; attempt++ {
		// Pass 1: find this key, or an empty bucket.
		//
		// Reclamation is deliberately NOT folded into this pass. Claiming a
		// dead bucket early in the chain while a live entry for the same key
		// sits later in it would create two entries for one key, and lookup --
		// which stops at the first match -- could then return the older one.
		for probe := uint64(0); probe < maxProbe; probe++ {
			i := (h + probe) & x.mask
			b := x.bucket(i)
			kw := fieldAt(b, offIdxKey)

			cur := ptrLoad64(kw)
			if cur == 0 {
				if ptrCAS64(kw, 0, key) {
					// The bucket is ours. Publish slot before ticket: a reader
					// gates on a non-zero ticket, so any reader that sees the
					// ticket is guaranteed to see the slot.
					ptrStore64(fieldAt(b, offIdxSlot), slot)
					ptrStore64(fieldAt(b, offIdxTicket), ticket)
					x.live.add(1)
					if probe > 0 {
						x.collides.add(probe)
					}
					return nil
				}
				// Lost the claim race; re-read before moving on, since the
				// winner may well have been inserting this very key.
				cur = ptrLoad64(kw)
			}

			if cur == key {
				x.updateMonotonic(b, ticket, slot)
				if probe > 0 {
					x.collides.add(probe)
				}
				return nil
			}
		}

		// Pass 2: the chain is full of other keys. Recycle the first dead one.
		//
		// Overwriting a bucket in place is safe for linear probing precisely
		// because it stays occupied: the chain is never broken, so every other
		// key's probe sequence still terminates where it did. Emptying a bucket
		// would break it, which is why reclamation replaces rather than clears.
		retry := false
		for probe := uint64(0); probe < maxProbe; probe++ {
			i := (h + probe) & x.mask
			b := x.bucket(i)
			kw := fieldAt(b, offIdxKey)

			cur := ptrLoad64(kw)
			if cur == 0 || cur == key {
				// The table changed under us: a bucket for this key may now
				// exist, so re-run pass 1 rather than guessing.
				retry = true
				break
			}
			if !x.isDead(b, cur, r) {
				continue
			}
			if ptrCAS64(kw, cur, key) {
				ptrStore64(fieldAt(b, offIdxSlot), slot)
				ptrStore64(fieldAt(b, offIdxTicket), ticket)
				x.reclaims.add(1)
				return nil
			}
		}
		if !retry {
			break
		}
	}

	x.overflows.add(1)
	return ErrIndexFull
}

// isDead reports whether a bucket's entry has been invalidated by the ring
// recycling the cell it points at.
//
// This is a hint, and it is allowed to be wrong in one direction only. If the
// entry's key is republished concurrently, a bucket judged dead here may be
// reclaimed a moment after it became live again; the consequence is that the
// key is briefly unfindable by Lookup until its next mutation, never that
// Lookup returns another key's data. The reverse error -- judging a dead bucket
// live -- costs nothing but a probe step.
//
//go:nosplit
func (x *keyIndex) isDead(b unsafe.Pointer, entryKey uint64, r *ring) bool {
	slot := ptrLoad64(fieldAt(b, offIdxSlot))
	if slot >= r.capacity {
		return true
	}
	c := cellAt(r.base, uintptr(slot), r.stride)
	return ptrLoad64(fieldAt(c, offKey)) != entryKey
}

// updateMonotonic advances a bucket's (ticket, slot) pair, but only forward.
//
// Producers on different cores commit out of order, so an update carrying an
// older ticket can arrive after a newer one. The CAS loop drops those instead of
// letting a stale mutation win. If the CAS fails, the value was moved by another
// core; re-read and re-test rather than retrying blindly, because the new value
// may already be newer than ours, in which case there is nothing left to do.
//
//go:nosplit
func (x *keyIndex) updateMonotonic(b unsafe.Pointer, ticket, slot uint64) {
	tw := fieldAt(b, offIdxTicket)
	for {
		cur := ptrLoad64(tw)
		if cur >= ticket {
			return // a newer mutation already won
		}
		if ptrCAS64(tw, cur, ticket) {
			ptrStore64(fieldAt(b, offIdxSlot), slot)
			return
		}
	}
}

// lookup resolves a key to its most recent record.
//
// The answer is verified against the physical cell before it is returned. Slots
// are recycled continuously by the ring, so between the index read and the cell
// read the cell may have been reused by an entirely different key. Comparing the
// cell's own key word is what turns that race from silent data corruption into
// an honest ErrNotFound.
//
// The returned Payload aliases arena memory and is valid only until that slot is
// republished, which the caller cannot prevent. Callers retaining the bytes must
// copy them.
//
//go:nosplit
func (x *keyIndex) lookup(key uint64, r *ring, rec *Record) error {
	if key == 0 {
		return ErrConfig
	}
	h := hashKey(key) & x.mask

	for probe := uint64(0); probe < maxProbe; probe++ {
		i := (h + probe) & x.mask
		b := x.bucket(i)

		cur := ptrLoad64(fieldAt(b, offIdxKey))
		if cur == 0 {
			return ErrNotFound // empty bucket terminates the probe sequence
		}
		if cur != key {
			continue
		}

		ticket := ptrLoad64(fieldAt(b, offIdxTicket))
		if ticket == 0 {
			return ErrNotFound // claimed but not yet populated
		}
		slot := ptrLoad64(fieldAt(b, offIdxSlot))
		if slot >= r.capacity {
			return ErrNotFound
		}

		// Read through to the authoritative cell.
		c := cellAt(r.base, uintptr(slot), r.stride)
		if ptrLoad64(fieldAt(c, offKey)) != key {
			return ErrNotFound // slot recycled by another key
		}
		if ptrLoad32(fieldAt(c, offFlags))&FlagCommitted == 0 {
			return ErrNotFound // slot is mid-rewrite
		}

		n := uintptr(ptrLoad64(fieldAt(c, offLength)))
		if n > r.payloadCap {
			return ErrNotFound // torn read against a cell being republished
		}

		rec.Ticket = ptrLoad64(fieldAt(c, offTicket))
		rec.Timestamp = int64(ptrLoad64(fieldAt(c, offTimestamp)))
		rec.Key = key
		rec.Epoch = ptrLoad64(fieldAt(c, offEpoch))
		rec.Checksum = ptrLoad32(fieldAt(c, offChecksum))
		rec.Flags = ptrLoad32(fieldAt(c, offFlags))
		rec.Slot = slot
		rec.Payload = payloadSlice(c, n)

		// Re-validate after the copy-out. If the key still matches, no producer
		// republished this cell underneath us and the record is coherent.
		if ptrLoad64(fieldAt(c, offKey)) != key {
			return ErrNotFound
		}
		return nil
	}
	return ErrNotFound
}

// IndexStats reports index health.
type IndexStats struct {
	Buckets    uint64
	Live       uint64
	LoadPct    uint64
	ProbeSteps uint64
	Overflows  uint64
	Reclaims   uint64
}

func (x *keyIndex) stats() IndexStats {
	live := x.live.load()
	return IndexStats{
		Buckets:    x.buckets,
		Live:       live,
		LoadPct:    live * 100 / x.buckets,
		ProbeSteps: x.collides.load(),
		Overflows:  x.overflows.load(),
		Reclaims:   x.reclaims.load(),
	}
}
