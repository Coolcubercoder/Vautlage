package vaultage

import (
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// Handler consumes one committed record.
//
// The Record and its Payload alias off-heap arena memory and are valid only
// until the handler returns. The instant it returns, the slot is released and a
// producer on another core may overwrite those exact bytes. A handler that needs
// to retain data must call rec.CopyPayload.
//
// The handler runs on a dedicated worker thread that owns its shard exclusively,
// so it is never invoked concurrently for the same shard.
type Handler func(rec *Record)

// Config describes the physical shape of an engine. Every field is fixed at
// construction: the engine performs no allocation, no growth and no rehashing
// after New returns.
type Config struct {
	// Shards is the number of independent matrices, each with its own arena,
	// ring, index and consumer thread. Rounded up to a power of two. Defaults
	// to GOMAXPROCS, which is the point at which per-core scaling saturates.
	Shards int

	// CellsPerShard is the ring capacity per shard. Must be a power of two.
	CellsPerShard uint64

	// PayloadCapacity is the usable payload bytes per cell. The cell stride
	// becomes 64+PayloadCapacity rounded up to a cache line, so choosing 448
	// yields an exact 512-byte stride and eight cells per 4KiB page.
	PayloadCapacity int

	// HighWaterPct is the saturation percentage at which the backpressure gate
	// latches shut. Defaults to 90.
	HighWaterPct uint64

	// LowWaterPct is the percentage at which the gate reopens. Defaults to 70.
	LowWaterPct uint64

	// IndexBucketsPerCell scales the key index relative to ring capacity.
	// Defaults to 2, i.e. a 50% maximum load factor, which keeps the linear
	// probe sequence short enough to stay inside a handful of cache lines.
	IndexBucketsPerCell uint64

	// RequireHugePages fails construction unless the kernel actually backs the
	// arena with huge pages. Leave false to accept a 2MiB-aligned mapping of
	// ordinary pages, which is the best available outcome on many hosts.
	RequireHugePages bool

	// RequirePinned fails construction if mlock(2) does not succeed. Set this
	// in production: an unpinned arena can be swapped, and a swapped matrix
	// turns a 100ns access into a multi-millisecond disk fault.
	RequirePinned bool

	// Checksums enables an FNV-1a digest over every payload on the write path.
	Checksums bool

	// PinWorkerThreads binds each consumer thread to a distinct logical CPU
	// (Linux only). Padding stops cores sharing a line; affinity stops one
	// worker migrating and dragging its working set through a cold cache.
	PinWorkerThreads bool

	// CoarseClock, when non-zero, replaces the per-record time.Now call with a
	// cached reading refreshed at this period by a background ticker. It roughly
	// halves publish latency at the cost of timestamp granularity. Record
	// ordering is unaffected, since ordering comes from the ticket.
	CoarseClock time.Duration

	// Consumer, when non-nil, starts one worker thread per shard. When nil the
	// engine is drained synchronously by the caller via Drain.
	Consumer Handler
}

// DefaultConfig returns a configuration sized for a general-purpose host:
// GOMAXPROCS shards of 65,536 cells at a 512-byte stride, which is 32MiB of
// pinned off-heap matrix per shard.
func DefaultConfig() Config {
	return Config{
		Shards:              runtime.GOMAXPROCS(0),
		CellsPerShard:       1 << 16,
		PayloadCapacity:     448,
		HighWaterPct:        90,
		LowWaterPct:         70,
		IndexBucketsPerCell: 2,
		RequireHugePages:    false,
		RequirePinned:       false,
		Checksums:           true,
		PinWorkerThreads:    true,
	}
}

func (c *Config) normalize() error {
	if c.Shards <= 0 {
		c.Shards = runtime.GOMAXPROCS(0)
	}
	// Round shards up to a power of two so key->shard routing is an AND rather
	// than a division on the hot path.
	n := 1
	for n < c.Shards {
		n <<= 1
	}
	c.Shards = n

	if c.CellsPerShard == 0 {
		c.CellsPerShard = 1 << 16
	}
	if !isPow2(uintptr(c.CellsPerShard)) {
		return ErrConfig
	}
	if c.PayloadCapacity < 0 {
		return ErrConfig
	}
	if c.PayloadCapacity == 0 {
		c.PayloadCapacity = 448
	}
	if c.HighWaterPct == 0 {
		c.HighWaterPct = 90
	}
	if c.LowWaterPct == 0 {
		c.LowWaterPct = 70
	}
	if c.IndexBucketsPerCell == 0 {
		c.IndexBucketsPerCell = 2
	}
	if !isPow2(uintptr(c.IndexBucketsPerCell)) {
		return ErrConfig
	}
	if c.HighWaterPct > 100 || c.LowWaterPct >= c.HighWaterPct {
		return ErrConfig
	}
	return nil
}

// ---------------------------------------------------------------------------
// CONTROL PLANE
//
// The shard control blocks live off-heap, in their own arena, for a reason that
// is easy to miss: a Go heap allocation is only guaranteed 8-byte alignment. If
// the shard structs sat on the heap, their padding would be *relative* padding
// with no fixed relationship to real cache lines, and two shards' hot cursors
// could still land in one line depending on the size class the allocator picked
// that day. Putting them in a 2MiB-aligned arena at a 128-byte-multiple stride
// makes the isolation exact and deterministic instead of probable.
//
// This is legal because neither shard nor globalCtl contains a single Go
// pointer: every field is an integer, an unsafe.Pointer into off-heap memory, or
// padding. The garbage collector therefore has nothing to trace here, which is
// exactly the property the engine is built around.
// ---------------------------------------------------------------------------

// globalCtl holds engine-wide hot state, each field on its own coherence stride.
type globalCtl struct {
	ticket  padU64 // global monotonic ticket dispenser
	running padU64 // 1 while the engine accepts ingestion
	epoch   padU64 // arena generation
	dropped padU64 // records rejected after a ticket was issued

	// saturated counts shards currently above the high-water mark. It is the
	// O(1) replacement for sweeping every shard on the hot path, and is touched
	// only when a shard crosses a threshold, never per record.
	saturated padU64

	// clock holds a coarse wall-clock reading in Unix nanoseconds, refreshed by
	// a background ticker when Config.CoarseClock is set.
	clock padU64
}

// shard is one independent matrix: arena span, ring, index, and counters.
type shard struct {
	ring  ring
	index keyIndex

	arenaBase unsafe.Pointer
	arenaSize uintptr
	id        uint64
	_         [CoherenceStride - 24]byte

	// hot is this shard's latched saturation flag: 1 once it crossed the
	// high-water mark, cleared only when it falls back below the low-water
	// mark. It is per-shard, on its own coherence stride, so testing and
	// flipping it never touches another core's line.
	hot      padU64
	ingested padU64
	consumed padU64
	rejected pad0U64
}

// pad0U64 is padU64 under a second name, used for the final field of a struct
// where the trailing padding also serves to round the struct size up.
type pad0U64 = padU64

var (
	// shardStride is sizeof(shard) rounded up to a coherence stride, so shard i
	// and shard i+1 can never share a cache line however the struct evolves.
	shardStride = alignUp(unsafe.Sizeof(shard{}), CoherenceStride)
	// ctlHeader is the offset of shard 0 within the control arena.
	ctlHeader = alignUp(unsafe.Sizeof(globalCtl{}), CoherenceStride)
)

const (
	// globalCtl must be exactly six coherence strides: six isolated cursors.
	_ = uint(unsafe.Sizeof(globalCtl{}) - 6*CoherenceStride)
	_ = uint(6*CoherenceStride - unsafe.Sizeof(globalCtl{}))
)

// Vaultage is the engine handle.
type Vaultage struct {
	cfg Config

	// Off-heap control plane.
	ctlArena *Arena
	ctl      *globalCtl
	shardPtr unsafe.Pointer // address of shard 0
	shardN   uint64
	shardMsk uint64

	// One data arena per shard.
	arenas []*Arena

	// The backpressure valve is engine-wide: saturation anywhere freezes
	// ingestion everywhere, because a fabric that keeps accepting writes it can
	// only route to one healthy shard is not bounded, merely lucky.
	valve gate

	payloadCap  uintptr
	cellStride  uintptr
	coarseClock bool

	wg      sync.WaitGroup
	stopCh  chan struct{}
	closeMu sync.Mutex
	closed  bool
}

// New constructs an engine: it maps and pins one arena per shard plus one for
// the control plane, lays the ring and index out inside each arena with exact
// pointer arithmetic, and starts one consumer thread per shard.
func New(cfg Config) (*Vaultage, error) {
	if err := cfg.normalize(); err != nil {
		return nil, err
	}

	v := &Vaultage{
		cfg:         cfg,
		shardN:      uint64(cfg.Shards),
		shardMsk:    uint64(cfg.Shards) - 1,
		payloadCap:  uintptr(cfg.PayloadCapacity),
		cellStride:  cellStrideFor(uintptr(cfg.PayloadCapacity)),
		coarseClock: cfg.CoarseClock > 0,
		stopCh:      make(chan struct{}),
	}
	if err := initGate(&v.valve, cfg.HighWaterPct, cfg.LowWaterPct); err != nil {
		return nil, err
	}

	// ---- control arena --------------------------------------------------
	ctlBytes := ctlHeader + shardStride*uintptr(cfg.Shards)
	ctlArena, err := NewArena(ctlBytes, false, cfg.RequirePinned)
	if err != nil {
		return nil, err
	}
	v.ctlArena = ctlArena
	v.ctl = (*globalCtl)(ctlArena.At(0))
	v.shardPtr = ctlArena.At(ctlHeader)

	v.ctl.ticket.store(0)
	v.ctl.epoch.store(uint64(time.Now().UnixNano()))
	v.ctl.dropped.store(0)
	v.ctl.saturated.store(0)
	v.ctl.clock.store(uint64(time.Now().UnixNano()))
	v.ctl.running.store(1)

	epoch := v.ctl.epoch.load()

	// ---- data arenas ----------------------------------------------------
	//
	// Each shard's arena holds the ring immediately followed by the index:
	//
	//   +0                          ring cells (cells * stride)
	//   +ringBytes                  index buckets (buckets * 32)
	//
	// ringBytes is a multiple of the cell stride, itself a multiple of 64, so
	// the index base inherits cache-line alignment from the 2MiB-aligned arena
	// base without any explicit padding.
	cells := cfg.CellsPerShard
	buckets := cells * cfg.IndexBucketsPerCell
	rb := ringBytes(cells, v.payloadCap)
	ib := indexBytes(buckets)

	v.arenas = make([]*Arena, 0, cfg.Shards)
	for i := 0; i < cfg.Shards; i++ {
		a, err := NewArena(rb+ib, cfg.RequireHugePages, cfg.RequirePinned)
		if err != nil {
			v.teardownArenas()
			_ = ctlArena.Close()
			return nil, err
		}
		v.arenas = append(v.arenas, a)

		sh := v.shard(i)
		sh.arenaBase = a.Base()
		sh.arenaSize = a.Size()
		sh.id = uint64(i)
		sh.hot.store(0)
		sh.ingested.store(0)
		sh.consumed.store(0)
		sh.rejected.store(0)

		if err := initRing(&sh.ring, a.At(0), cells, v.payloadCap, epoch); err != nil {
			v.teardownArenas()
			_ = ctlArena.Close()
			return nil, err
		}
		if err := initIndex(&sh.index, a.At(rb), buckets); err != nil {
			v.teardownArenas()
			_ = ctlArena.Close()
			return nil, err
		}
	}

	// ---- background clock ------------------------------------------------
	if v.coarseClock {
		v.wg.Add(1)
		go v.clockTicker(cfg.CoarseClock)
	}

	// ---- consumer threads ------------------------------------------------
	if cfg.Consumer != nil {
		for i := 0; i < cfg.Shards; i++ {
			v.wg.Add(1)
			go v.worker(i, cfg.Consumer)
		}
	}
	return v, nil
}

// shard returns the control block for shard i.
//
//	addr = shardBase + i*shardStride
//
//go:nosplit
func (v *Vaultage) shard(i int) *shard {
	return (*shard)(unsafe.Add(v.shardPtr, uintptr(i)*shardStride))
}

// route maps a key to its shard. Keys are avalanched first: routing on raw key
// bits would send dense or structured keys to a handful of shards and leave the
// rest of the machine idle.
//
//go:nosplit
func (v *Vaultage) route(key uint64) uint64 {
	return hashKey(key) & v.shardMsk
}

// Publish ingests one state mutation and returns its ticket.
//
// The whole path is: one atomic load of the gate, one atomic add for the ticket,
// one CAS to claim a slot, a memmove of the payload, seven plain stores, one
// atomic release store, and one CAS into the index. There is no lock, no
// channel, no interface dispatch and no heap allocation anywhere in it.
func (v *Vaultage) Publish(key uint64, payload []byte) (uint64, error) {
	return v.publish(key, payload, 0)
}

// PublishTombstone records a logical deletion of key.
func (v *Vaultage) PublishTombstone(key uint64) (uint64, error) {
	return v.publish(key, nil, FlagTombstone)
}

func (v *Vaultage) publish(key uint64, payload []byte, flags uint32) (uint64, error) {
	// Key 0 is the index's empty sentinel and cannot be stored.
	if key == 0 {
		return 0, ErrConfig
	}
	if uintptr(len(payload)) > v.payloadCap {
		return 0, ErrPayloadTooLarge
	}
	if v.ctl.running.load() == 0 {
		return 0, ErrClosed
	}

	// Backpressure is checked before the ticket is issued, so a frozen engine
	// costs a producer exactly one cache-hot atomic load.
	if v.valve.isClosed() {
		v.valve.reject()
		return 0, ErrBackpressure
	}

	sid := v.route(key)
	sh := v.shard(int(sid))

	ticket := v.ctl.ticket.add(1)

	pos, ok := sh.ring.claim()
	if !ok {
		// This shard filled between the gate check and the claim. Re-evaluate
		// saturation immediately so the gate latches rather than letting every
		// subsequent producer discover fullness the expensive way.
		sh.rejected.add(1)
		v.ctl.dropped.add(1)
		v.markSaturated(sh)
		return 0, ErrShardFull
	}

	var sum uint32
	if v.cfg.Checksums {
		sum = checksum32(payload)
	}

	sh.ring.publish(pos, ticket, key, v.now(), flags, payload, sum)
	sh.ingested.add(1)

	// Point the index at the new record. An overflowing index is not fatal: the
	// record is committed and will still be delivered to the consumer in ticket
	// order, it simply is not addressable by key lookup.
	_ = sh.index.upsert(key, ticket, pos&sh.ring.mask, &sh.ring)

	// O(1) saturation test against this shard's own, already-cached cursors.
	if sh.ring.occupancyPct() >= v.cfg.HighWaterPct {
		v.markSaturated(sh)
	}
	return ticket, nil
}

// Lookup resolves a key to its most recent record, reading straight out of
// off-heap memory with no copy and no allocation.
//
// The returned Payload aliases a live cell that producers may recycle at any
// moment. Callers retaining the bytes must call CopyPayload.
func (v *Vaultage) Lookup(key uint64, rec *Record) error {
	if key == 0 {
		return ErrConfig
	}
	sid := v.route(key)
	sh := v.shard(int(sid))
	if err := sh.index.lookup(key, &sh.ring, rec); err != nil {
		return err
	}
	rec.Shard = int(sid)
	return nil
}

// ---------------------------------------------------------------------------
// SATURATION AND BACKPRESSURE
// ---------------------------------------------------------------------------

// saturation returns aggregate and worst-shard occupancy, both as percentages.
//
// This walks every shard, so it is a reporting function only. It is called by
// Stats and never by Publish: reading every shard's cursors on the write path
// would mean every producer pulling every other shard's cache lines on every
// record, which is a true-sharing storm that makes throughput fall as cores are
// added. The hot path uses the O(1) latch below instead.
func (v *Vaultage) saturation() (aggPct, maxPct uint64) {
	var pending, capacity uint64
	for i := uint64(0); i < v.shardN; i++ {
		sh := v.shard(int(i))
		p := sh.ring.pending()
		pending += p
		capacity += sh.ring.capacity
		if pct := p * 100 / sh.ring.capacity; pct > maxPct {
			maxPct = pct
		}
	}
	if capacity == 0 {
		return 0, 0
	}
	return pending * 100 / capacity, maxPct
}

// markSaturated latches a shard hot and freezes ingestion engine-wide.
//
// The cost model is what matters here. The common case is a producer reading
// its own shard's occupancy from a line it already owns and finding it below the
// high-water mark: two loads, no atomic write, no foreign cache line. Only the
// single producer that observes the *transition* wins the per-shard CAS, and
// only that one touches the global counter and the gate.
//
// Freezing engine-wide on a single hot shard is deliberate. Keys route to shards
// by hash, so a caller cannot steer writes away from a saturated shard; an
// engine that kept accepting writes because *other* shards had room would be
// bounded only by luck.
//
//go:nosplit
func (v *Vaultage) markSaturated(sh *shard) {
	if sh.hot.cas(0, 1) {
		v.ctl.saturated.add(1)
		v.valve.shut()
	}
}

// markDrained clears a shard's hot latch once it falls below the low-water mark,
// and reopens the gate when no shard remains saturated.
//
// The gap between the two thresholds is the hysteresis that stops the valve
// flapping: a shard that has just been throttled must give back a fifth of its
// capacity before it counts as healthy again.
//
//go:nosplit
func (v *Vaultage) markDrained(sh *shard) {
	if sh.hot.load() == 0 {
		return // fast path: this shard was never throttled
	}
	if sh.ring.occupancyPct() > v.cfg.LowWaterPct {
		return // still above the low-water mark; stay latched
	}
	if !sh.hot.cas(1, 0) {
		return // another consumer cleared it first
	}
	if v.ctl.saturated.add(^uint64(0)) == 0 {
		v.valve.release()
		// A shard may have gone hot between the decrement and the release. Re-
		// test and re-shut so the gate cannot be left open over a hot shard.
		if v.ctl.saturated.load() != 0 {
			v.valve.shut()
		}
	}
}

// now returns the commit timestamp.
//
// time.Now costs ~27ns on this class of hardware, which measured as more than
// half of the entire write path. When Config.CoarseClock is set, producers read
// a cached value refreshed by a background ticker instead, trading timestamp
// granularity for roughly half the publish latency. Record *ordering* never
// depends on this value -- that is what the ticket is for -- so a coarse clock
// costs only metadata precision.
//
//go:nosplit
func (v *Vaultage) now() int64 {
	if v.coarseClock {
		return int64(v.ctl.clock.load())
	}
	return time.Now().UnixNano()
}

// clockTicker refreshes the coarse clock until the engine stops.
func (v *Vaultage) clockTicker(period time.Duration) {
	defer v.wg.Done()
	t := time.NewTicker(period)
	defer t.Stop()
	for {
		select {
		case <-v.stopCh:
			// Explicit stop signal rather than polling `running` on the tick:
			// otherwise Close would block for up to one full clock period.
			return
		case <-t.C:
			v.ctl.clock.store(uint64(time.Now().UnixNano()))
		}
	}
}

// ---------------------------------------------------------------------------
// CONSUMPTION
// ---------------------------------------------------------------------------

// drainBatch is the number of records a worker consumes before it re-checks the
// gate. Batching amortises the saturation sweep across many records while
// staying small enough that backpressure is released promptly.
const drainBatch = 256

// worker is the per-shard consumer thread.
//
// It owns its shard exclusively: one consumer per ring means the head cursor is
// uncontended, and because every shard's cursors sit on their own coherence
// strides in a separate 2MiB-aligned arena, two workers never touch the same
// physical cache line. That is the property that makes aggregate throughput
// scale with core count instead of collapsing into coherence traffic.
func (v *Vaultage) worker(id int, h Handler) {
	defer v.wg.Done()

	// Bind the goroutine to one OS thread for the engine's lifetime, then bind
	// that thread to a core. Without LockOSThread the scheduler could move this
	// goroutine between threads and undo the affinity on every preemption.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if v.cfg.PinWorkerThreads {
		_ = sysPinThreadToCore(id % runtime.NumCPU())
	}

	sh := v.shard(id)
	var rec Record
	rec.Shard = id
	idle := 0

	for {
		n := 0
		for ; n < drainBatch; n++ {
			pos, ok := sh.ring.dequeue(&rec)
			if !ok {
				break
			}
			rec.Shard = id
			h(&rec)
			// Release only after the handler returns: until then the payload
			// slice it holds still aliases this cell.
			sh.ring.release(pos)
			sh.consumed.add(1)
		}

		if n > 0 {
			idle = 0
			v.markDrained(sh)
			continue
		}

		// Ring empty. Exit only once ingestion has stopped and nothing remains
		// in flight, including slots claimed but not yet committed.
		if v.ctl.running.load() == 0 && sh.ring.pending() == 0 {
			return
		}
		idle = backoff(idle)
	}
}

// backoff is the idle ladder for an empty ring.
//
// The first tier does nothing at all, deliberately. The caller's next action is
// another dequeue, which reads the commit cursor from cache, so "spinning" here
// means re-reading one hot line with no call, no syscall and no scheduler
// interaction. At high ingest rates the next record is tens of nanoseconds away
// and any form of parking costs orders of magnitude more than the wait itself.
// Go's asynchronous preemption keeps this loop safe even at GOMAXPROCS=1.
//
// Past that, a consumer that keeps burning a core would starve the very
// producers it is waiting for, so the ladder escalates to a scheduler yield and
// then to real sleeps. The final tier clamps the counter so it cannot overflow
// across a long idle period.
func backoff(idle int) int {
	switch {
	case idle < 32:
		// Tier 1: pure retry. Re-read the commit cursor and try again.
	case idle < 1024:
		runtime.Gosched()
	case idle < 4096:
		time.Sleep(50 * time.Microsecond)
	default:
		time.Sleep(500 * time.Microsecond)
		return idle
	}
	return idle + 1
}

// drainPool supplies the scratch Record used by Drain. See the comment in Drain
// for why it cannot simply be a local variable.
var drainPool = sync.Pool{New: func() any { return new(Record) }}

// Drain synchronously consumes up to max records from every shard on the
// calling goroutine, and returns how many were delivered. It is the manual
// alternative to a Consumer, intended for tests, single-threaded embeddings and
// deterministic replay. Passing max <= 0 drains everything currently committed.
func (v *Vaultage) Drain(max int, h Handler) int {
	// The scratch Record is pooled rather than stack-allocated. Passing its
	// address to an opaque Handler forces it to escape, and a per-call heap
	// allocation on the drain path would contradict the engine's central claim.
	// The pool makes steady-state Drain genuinely allocation-free while keeping
	// it safe to call from several goroutines at once.
	rp := drainPool.Get().(*Record)
	defer drainPool.Put(rp)
	rec := rp
	total := 0
	for i := uint64(0); i < v.shardN; i++ {
		sh := v.shard(int(i))
		for max <= 0 || total < max {
			pos, ok := sh.ring.dequeue(rec)
			if !ok {
				break
			}
			rec.Shard = int(i)
			if h != nil {
				h(rec)
			}
			sh.ring.release(pos)
			sh.consumed.add(1)
			total++
		}
		v.markDrained(sh)
	}
	return total
}

// Close stops ingestion, drains every shard, joins the consumer threads, then
// unpins and unmaps every arena.
//
// Ordering here is not negotiable. Records in flight alias arena memory, so the
// mapping may only be torn down after the last worker has returned; unmapping
// first would turn an in-flight handler into a SIGSEGV rather than a Go panic.
func (v *Vaultage) Close() error {
	v.closeMu.Lock()
	defer v.closeMu.Unlock()
	if v.closed {
		return nil
	}
	v.closed = true

	// Stop accepting new work, then release the valve so a worker cannot be
	// waiting on a gate that nothing will ever reopen.
	v.ctl.running.store(0)
	v.valve.forceOpen()
	close(v.stopCh)

	// Workers exit on their own once running is clear and their ring is empty.
	v.wg.Wait()

	// With no consumer configured, whatever is still committed is dropped; the
	// slots are about to be unmapped regardless.
	var firstErr error
	for _, a := range v.arenas {
		if err := a.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	v.arenas = nil
	if err := v.ctlArena.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	// Null the control pointers: any later use is now a clean nil dereference
	// rather than a read of unmapped memory.
	v.ctl, v.shardPtr, v.ctlArena = nil, nil, nil
	return firstErr
}

// teardownArenas releases data arenas during a failed construction.
func (v *Vaultage) teardownArenas() {
	for _, a := range v.arenas {
		_ = a.Close()
	}
	v.arenas = nil
}
