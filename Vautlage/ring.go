package vaultage

import "unsafe"

// ring is a bounded, multi-producer/multi-consumer, lock-free sequence ring
// whose cells live in off-heap arena memory.
//
// # PROTOCOL
//
// Each cell carries a `seq` word at offset 0 that encodes the cell's state as a
// position rather than as a flag. For a ring of capacity C and a slot index
// i = pos & (C-1):
//
//	seq == pos          -> the cell is free and belongs to the producer whose
//	                       ticket position is pos. It is the producer's turn.
//	seq == pos+1        -> the producer has finished writing; the payload is
//	                       published and the cell belongs to a consumer.
//	seq == pos+C        -> the consumer has finished reading; the cell is free
//	                       again for the *next* lap's producer.
//
// Encoding state as a monotonically increasing position is what makes the ring
// ABA-immune without a tag word or hazard pointers: a stale CAS attempt from a
// descheduled thread carries an old position that can never again match, because
// seq only ever moves forward by one lap at a time.
//
// CURSORS
//
//	tail   producer cursor: the next position to be handed out.
//	commit contiguous commit index: every position below it is fully written.
//	head   consumer cursor: the next position to be consumed.
//
// The invariant head <= commit <= tail holds at all times. Consumers are gated
// on `commit` rather than on `tail`, so a consumer can never observe a slot that
// a producer has claimed but not yet finished writing, and records are delivered
// in exact ticket order. That is what makes the stream replicable: `commit` is
// the durable prefix of the log.
//
// Each cursor occupies its own coherence stride. Without that padding, every
// producer CAS on `tail` would invalidate the consumers' cached copy of `head`
// and vice versa, and measured throughput would *fall* as cores were added.
type ring struct {
	// --- immutable after construction, read by every core on every operation.
	// Grouped together so they share clean, never-invalidated lines.
	base       unsafe.Pointer // address of cell 0 inside the arena
	stride     uintptr        // bytes between consecutive cells
	capacity   uint64         // number of cells; always a power of two
	mask       uint64         // capacity-1; replaces a modulo with an AND
	payloadCap uintptr        // usable payload bytes per cell
	epoch      uint64         // arena generation stamped into every record
	_          [CoherenceStride - 48]byte

	// --- hot cursors, one per coherence stride.
	tail   padU64 // producer ticket dispenser
	commit padU64 // contiguous committed prefix
	head   padU64 // consumer cursor
}

// Compile-time assertion that the hot cursors really are on separate strides.
// If a field is ever added to the immutable block without shrinking the padding,
// these fail the build rather than silently reintroducing false sharing.
const (
	_ = uint(unsafe.Offsetof(ring{}.commit) - unsafe.Offsetof(ring{}.tail) - CoherenceStride)
	_ = uint(CoherenceStride - (unsafe.Offsetof(ring{}.commit) - unsafe.Offsetof(ring{}.tail)))
	_ = uint(unsafe.Offsetof(ring{}.head) - unsafe.Offsetof(ring{}.commit) - CoherenceStride)
	_ = uint(CoherenceStride - (unsafe.Offsetof(ring{}.head) - unsafe.Offsetof(ring{}.commit)))

	// The cursor block must begin on a stride boundary so the padding is
	// meaningful with respect to real cache lines.
	_ = uint(0 - unsafe.Offsetof(ring{}.tail)%CoherenceStride)
)

// ringBytes returns the exact number of arena bytes a ring of `cells` cells with
// `payloadCap` payload bytes per cell will occupy.
func ringBytes(cells uint64, payloadCap uintptr) uintptr {
	return uintptr(cells) * cellStrideFor(payloadCap)
}

// initRing binds a ring to a region of arena memory and primes every cell's
// sequence word.
//
// `base` must be at least 64-byte aligned; in practice it is derived from a
// 2MiB-aligned arena base plus a stride-multiple offset, so it is.
func initRing(r *ring, base unsafe.Pointer, cells uint64, payloadCap uintptr, epoch uint64) error {
	if cells == 0 || !isPow2(uintptr(cells)) {
		return ErrConfig
	}
	if !isAligned(uintptr(base), CacheLineSize) {
		return ErrConfig
	}

	r.base = base
	r.stride = cellStrideFor(payloadCap)
	r.capacity = cells
	r.mask = cells - 1
	r.payloadCap = payloadCap
	r.epoch = epoch

	// Prime the turnstile: cell i starts life owned by the producer at position
	// i. This is the initial condition that makes `seq == pos` mean "free".
	for i := uint64(0); i < cells; i++ {
		c := cellAt(base, uintptr(i), r.stride)
		ptrStore64(fieldAt(c, offSeq), i)
		ptrStore64(fieldAt(c, offTicket), 0)
		ptrStore64(fieldAt(c, offTimestamp), 0)
		ptrStore64(fieldAt(c, offKey), 0)
		ptrStore64(fieldAt(c, offLength), 0)
		ptrStore64(fieldAt(c, offEpoch), epoch)
		ptrStore64(fieldAt(c, offReserved), 0)
		ptrStore32(fieldAt(c, offChecksum), 0)
		ptrStore32(fieldAt(c, offFlags), 0)
	}

	r.tail.store(0)
	r.commit.store(0)
	r.head.store(0)
	return nil
}

// cell returns the address of the cell serving ring position `pos`.
//
//	addr = base + (pos & mask) * stride
//
// The mask is what makes this two instructions instead of a division: capacity
// is a power of two, so the wrap is a bitwise AND.
//
//go:nosplit
func (r *ring) cell(pos uint64) unsafe.Pointer {
	return cellAt(r.base, uintptr(pos&r.mask), r.stride)
}

// claim acquires exclusive ownership of the next ring position.
//
// This is the write-path turnstile. The loop is deliberately un-abstracted: a
// load of the cursor, a load of the cell's sequence word, a signed comparison,
// and a single CompareAndSwapUint64. There is no lock, no channel, no closure,
// and no allocation anywhere in it.
//
// The signed difference is the whole algorithm:
//
//	dif == 0  the cell is free and it is our turn -> try to take the ticket.
//	dif <  0  the cell still holds an unconsumed record from the previous lap,
//	          i.e. the ring is full. Report saturation instead of spinning; the
//	          caller converts that into backpressure.
//	dif >  0  another producer won the race and already advanced tail; our view
//	          of the cursor is stale, so reload and retry.
//
// Returns the claimed position; ok is false when the ring is saturated.
//
//go:nosplit
func (r *ring) claim() (pos uint64, ok bool) {
	for {
		pos = r.tail.load()
		c := r.cell(pos)
		seq := ptrLoad64(fieldAt(c, offSeq))

		// Signed arithmetic on wrapped counters: the difference stays correct
		// across a uint64 overflow, which at 10^9 tickets/sec is ~584 years
		// away but costs nothing to handle correctly.
		dif := int64(seq) - int64(pos)

		switch {
		case dif == 0:
			if r.tail.cas(pos, pos+1) {
				return pos, true
			}
			// Lost the race; another producer took this position. Retry.
		case dif < 0:
			return 0, false // ring saturated
		default:
			// Stale cursor read; fall through and reload.
		}
	}
}

// publish writes a record into a claimed slot and releases it to consumers.
//
// Ordering is the point of this function. Every header field and every payload
// byte is written with a plain store, and only then is the sequence word
// published with an atomic store. In the Go memory model that atomic store is
// sequentially consistent, so any core that observes seq == pos+1 is guaranteed
// to observe every preceding write in this function. No explicit fence
// intrinsic is required, and none exists in portable Go.
//
//go:nosplit
func (r *ring) publish(pos, ticket, key uint64, ts int64, flags uint32, payload []byte, checksum uint32) uintptr {
	c := r.cell(pos)

	n := copyIntoPayload(c, payload)

	ptrStore64(fieldAt(c, offTicket), ticket)
	ptrStore64(fieldAt(c, offTimestamp), uint64(ts))
	ptrStore64(fieldAt(c, offKey), key)
	ptrStore64(fieldAt(c, offLength), uint64(n))
	ptrStore64(fieldAt(c, offEpoch), r.epoch)
	ptrStore32(fieldAt(c, offChecksum), checksum)
	ptrStore32(fieldAt(c, offFlags), flags|FlagCommitted)

	// Release edge. After this store the slot is no longer ours.
	ptrStore64(fieldAt(c, offSeq), pos+1)

	// Drag the contiguous commit index forward over this and any other slots
	// that finished out of order.
	r.advanceCommit()
	return n
}

// advanceCommit walks the commit index forward across every contiguously
// published position.
//
// Producers finish out of order: a producer holding position 7 may publish
// before the one holding position 5. `commit` may therefore only advance when
// the slot it points at has actually been published, which is exactly the
// condition seq == commit+1.
//
// Every producer runs this helper loop, so no single thread owns the job of
// advancing the index. If the producer that would have advanced it is
// descheduled mid-loop, the next producer through completes the work. The index
// is never left behind by a sleeping thread.
//
//go:nosplit
func (r *ring) advanceCommit() {
	for {
		c := r.commit.load()
		if c == r.tail.load() {
			return // nothing has been claimed beyond the commit point
		}
		cell := r.cell(c)
		if ptrLoad64(fieldAt(cell, offSeq)) != c+1 {
			// Position c is claimed but its producer has not published yet.
			// The commit prefix genuinely stops here.
			return
		}
		if !r.commit.cas(c, c+1) {
			continue // another producer advanced it; re-read and keep helping
		}
	}
}

// dequeue consumes the next committed record in ticket order.
//
// This is the read-path turnstile. Consumers race only on `head`, and are gated
// by `commit`, so a consumer can never see a half-written cell: the gate is a
// data dependency, not a lock.
//
// The returned Record's Payload aliases arena memory directly. It stays valid
// only until release is called for that slot, which is why dequeue hands back
// the position and the caller must call release when finished. Nothing is
// copied and nothing is allocated.
//
//go:nosplit
func (r *ring) dequeue(rec *Record) (pos uint64, ok bool) {
	for {
		pos = r.head.load()
		if pos >= r.commit.load() {
			return 0, false // nothing committed to consume
		}
		if !r.head.cas(pos, pos+1) {
			continue // another consumer took this position
		}

		c := r.cell(pos)

		// Acquire edge: pairs with the producer's release store in publish and
		// re-confirms that this cell belongs to this lap.
		if ptrLoad64(fieldAt(c, offSeq)) != pos+1 {
			// Structurally unreachable: pos < commit implies the slot was
			// published. Treated as a hard invariant violation rather than
			// silently returning corrupt data.
			r.release(pos)
			return 0, false
		}

		n := uintptr(ptrLoad64(fieldAt(c, offLength)))
		rec.Ticket = ptrLoad64(fieldAt(c, offTicket))
		rec.Timestamp = int64(ptrLoad64(fieldAt(c, offTimestamp)))
		rec.Key = ptrLoad64(fieldAt(c, offKey))
		rec.Epoch = ptrLoad64(fieldAt(c, offEpoch))
		rec.Checksum = ptrLoad32(fieldAt(c, offChecksum))
		rec.Flags = ptrLoad32(fieldAt(c, offFlags))
		rec.Slot = pos & r.mask
		rec.Payload = payloadSlice(c, n)
		return pos, true
	}
}

// release returns a consumed slot to the producers.
//
// The new sequence value is pos+capacity: precisely the position of the
// producer who will own this cell on the next lap. That single store is what
// makes the free-list implicit; there is no free list, no bitmap and no
// allocator, just an arithmetic identity.
//
//go:nosplit
func (r *ring) release(pos uint64) {
	c := r.cell(pos)
	ptrStore64(fieldAt(c, offSeq), pos+r.capacity)
}

// pending returns the number of positions claimed but not yet consumed. This is
// the ring's occupancy and the input to the backpressure gate.
//
//go:nosplit
func (r *ring) pending() uint64 {
	// Load head first. If a concurrent consumer advances between the two loads
	// the result is a slight over-estimate, which is the safe direction for a
	// saturation signal: it can trip backpressure a hair early, never late.
	h := r.head.load()
	t := r.tail.load()
	if t < h {
		return 0
	}
	return t - h
}

// committed returns the number of records published and awaiting consumption.
//
//go:nosplit
func (r *ring) committed() uint64 {
	h := r.head.load()
	c := r.commit.load()
	if c < h {
		return 0
	}
	return c - h
}

// occupancyPct returns ring occupancy scaled to 0..100.
//
//go:nosplit
func (r *ring) occupancyPct() uint64 {
	return r.pending() * 100 / r.capacity
}
