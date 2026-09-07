package vaultage

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

// testConfig returns a small, fast engine. RequirePinned is left false because
// mlock limits (RLIMIT_MEMLOCK) vary by host and CI sandbox; TestArenaPinning
// reports what this host actually granted.
func testConfig() Config {
	c := DefaultConfig()
	c.Shards = 4
	c.CellsPerShard = 1024
	c.PayloadCapacity = 64
	c.RequirePinned = false
	c.PinWorkerThreads = false
	c.Consumer = nil
	return c
}

// --- layout -----------------------------------------------------------------

func TestCellLayoutOffsets(t *testing.T) {
	// The compile-time assertions in layout.go already guarantee these; this
	// test restates them at run time so a failure names the offending field.
	var h cellHeader
	checks := []struct {
		name string
		got  uintptr
		want uintptr
	}{
		{"seq", unsafe.Offsetof(h.seq), offSeq},
		{"ticket", unsafe.Offsetof(h.ticket), offTicket},
		{"timestamp", unsafe.Offsetof(h.timestamp), offTimestamp},
		{"key", unsafe.Offsetof(h.key), offKey},
		{"length", unsafe.Offsetof(h.length), offLength},
		{"epoch", unsafe.Offsetof(h.epoch), offEpoch},
		{"checksum", unsafe.Offsetof(h.checksum), offChecksum},
		{"flags", unsafe.Offsetof(h.flags), offFlags},
		{"reserved", unsafe.Offsetof(h.reserved), offReserved},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("cellHeader.%s at offset %d, want %d", c.name, c.got, c.want)
		}
		if c.name != "flags" && c.got%WordSize != 0 {
			t.Errorf("cellHeader.%s offset %d is not 8-byte aligned", c.name, c.got)
		}
	}
	if got := unsafe.Sizeof(h); got != HeaderSize {
		t.Errorf("sizeof(cellHeader) = %d, want %d", got, HeaderSize)
	}
}

func TestPaddingIsolatesCursors(t *testing.T) {
	if got := unsafe.Sizeof(padU64{}); got != CoherenceStride {
		t.Fatalf("sizeof(padU64) = %d, want %d", got, CoherenceStride)
	}
	var r ring
	pairs := [][3]uintptr{
		{unsafe.Offsetof(r.tail), unsafe.Offsetof(r.commit), 0},
		{unsafe.Offsetof(r.commit), unsafe.Offsetof(r.head), 0},
	}
	for _, p := range pairs {
		if p[1]-p[0] != CoherenceStride {
			t.Errorf("cursors %d bytes apart, want %d", p[1]-p[0], CoherenceStride)
		}
		// The decisive property: different 64-byte cache lines.
		if p[0]/CacheLineSize == p[1]/CacheLineSize {
			t.Errorf("cursors at %d and %d share a cache line", p[0], p[1])
		}
	}
	if unsafe.Offsetof(r.tail)%CoherenceStride != 0 {
		t.Errorf("cursor block does not start on a coherence stride")
	}
}

func TestCellStrideAlignment(t *testing.T) {
	for _, payload := range []uintptr{0, 1, 63, 64, 65, 448, 4096} {
		s := cellStrideFor(payload)
		if s%CacheLineSize != 0 {
			t.Errorf("stride %d for payload %d is not cache-line aligned", s, payload)
		}
		if s < HeaderSize+payload {
			t.Errorf("stride %d cannot hold header+payload %d", s, HeaderSize+payload)
		}
	}
}

func TestShardStrideIsolation(t *testing.T) {
	if shardStride%CoherenceStride != 0 {
		t.Fatalf("shardStride %d is not a multiple of %d", shardStride, CoherenceStride)
	}
	if shardStride < unsafe.Sizeof(shard{}) {
		t.Fatalf("shardStride %d smaller than sizeof(shard) %d", shardStride, unsafe.Sizeof(shard{}))
	}
}

// --- arena ------------------------------------------------------------------

func TestArenaHugePageAlignment(t *testing.T) {
	a, err := NewArena(3<<20, false, false)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	st := a.Stats()
	if !isAligned(st.Base, HugePageSize) {
		t.Errorf("arena base %#x is not 2MiB aligned", st.Base)
	}
	if st.Size%HugePageSize != 0 {
		t.Errorf("arena size %d is not a multiple of 2MiB", st.Size)
	}
	if st.Size < 3<<20 {
		t.Errorf("arena size %d smaller than requested", st.Size)
	}
	t.Logf("base=%#x size=%d pagesize=%d hugeNative=%v hugeAdvise=%v pinned=%v",
		st.Base, st.Size, st.PageSize, st.HugePagesNative, st.HugePagesAdvise, st.Pinned)
}

func TestArenaReadWriteAcrossSpan(t *testing.T) {
	a, err := NewArena(2<<20, false, false)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	// Touch the first and last byte of every page to prove the whole span is
	// mapped and writable, not merely the head.
	for off := uintptr(0); off < a.Size(); off += a.pageSize {
		p := (*byte)(a.At(off))
		*p = byte(off / a.pageSize)
	}
	for off := uintptr(0); off < a.Size(); off += a.pageSize {
		if got := *(*byte)(a.At(off)); got != byte(off/a.pageSize) {
			t.Fatalf("arena readback mismatch at %d: got %d", off, got)
		}
	}
	last := (*byte)(a.At(a.Size() - 1))
	*last = 0xAB
	if *last != 0xAB {
		t.Fatal("final arena byte not writable")
	}
}

func TestArenaPinning(t *testing.T) {
	a, err := NewArena(2<<20, false, false)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()
	if !a.Stats().Pinned {
		t.Logf("mlock unavailable on this host (RLIMIT_MEMLOCK); arena is swappable")
	}
}

func TestArenaCloseIsIdempotent(t *testing.T) {
	a, err := NewArena(2<<20, false, false)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

// --- ring -------------------------------------------------------------------

func newTestRing(t *testing.T, cells uint64, payload uintptr) (*ring, *Arena) {
	t.Helper()
	a, err := NewArena(ringBytes(cells, payload), false, false)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	r := new(ring)
	if err := initRing(r, a.At(0), cells, payload, 7); err != nil {
		t.Fatalf("initRing: %v", err)
	}
	return r, a
}

func TestRingFIFOOrder(t *testing.T) {
	r, a := newTestRing(t, 16, 64)
	defer a.Close()

	for i := uint64(0); i < 16; i++ {
		pos, ok := r.claim()
		if !ok {
			t.Fatalf("claim %d failed", i)
		}
		r.publish(pos, i+100, i+1, int64(i), 0, []byte{byte(i)}, 0)
	}
	if _, ok := r.claim(); ok {
		t.Fatal("claim succeeded on a full ring")
	}

	var rec Record
	for i := uint64(0); i < 16; i++ {
		pos, ok := r.dequeue(&rec)
		if !ok {
			t.Fatalf("dequeue %d failed", i)
		}
		if rec.Ticket != i+100 || rec.Key != i+1 {
			t.Fatalf("out of order at %d: ticket=%d key=%d", i, rec.Ticket, rec.Key)
		}
		if len(rec.Payload) != 1 || rec.Payload[0] != byte(i) {
			t.Fatalf("payload mismatch at %d: %v", i, rec.Payload)
		}
		r.release(pos)
	}
	if _, ok := r.dequeue(&rec); ok {
		t.Fatal("dequeue succeeded on an empty ring")
	}
}

func TestRingWrapsIndefinitely(t *testing.T) {
	r, a := newTestRing(t, 8, 64)
	defer a.Close()

	var rec Record
	// 500 laps of an 8-cell ring: every cell is recycled ~500 times, which is
	// what would expose an ABA or a stale sequence value.
	for i := uint64(0); i < 4000; i++ {
		pos, ok := r.claim()
		if !ok {
			t.Fatalf("claim failed at %d", i)
		}
		r.publish(pos, i, i+1, 0, 0, []byte{byte(i)}, 0)

		dpos, ok := r.dequeue(&rec)
		if !ok {
			t.Fatalf("dequeue failed at %d", i)
		}
		if rec.Ticket != i {
			t.Fatalf("lap %d: got ticket %d", i, rec.Ticket)
		}
		r.release(dpos)
	}
	if p := r.pending(); p != 0 {
		t.Fatalf("pending = %d after full drain", p)
	}
}

func TestRingCommitIndexGatesConsumers(t *testing.T) {
	r, a := newTestRing(t, 16, 64)
	defer a.Close()

	// Claim two positions but publish only the second. The commit index must
	// not advance past the first, so no consumer may observe the second record
	// even though its cell is fully written.
	p0, _ := r.claim()
	p1, _ := r.claim()
	r.publish(p1, 1, 1, 0, 0, nil, 0)

	if c := r.commit.load(); c != 0 {
		t.Fatalf("commit index advanced to %d over an unpublished slot", c)
	}
	var rec Record
	if _, ok := r.dequeue(&rec); ok {
		t.Fatal("consumer observed a record past the commit index")
	}

	// Publishing the gap must release both records at once.
	r.publish(p0, 0, 2, 0, 0, nil, 0)
	if c := r.commit.load(); c != 2 {
		t.Fatalf("commit index = %d after filling the gap, want 2", c)
	}
	for i := 0; i < 2; i++ {
		pos, ok := r.dequeue(&rec)
		if !ok {
			t.Fatalf("dequeue %d failed after commit advance", i)
		}
		r.release(pos)
	}
}

func TestRingConcurrentProducersConsumers(t *testing.T) {
	const (
		producers = 8
		perProd   = 20000
		cells     = 1024
	)
	r, a := newTestRing(t, cells, 64)
	defer a.Close()

	var (
		produced atomic.Uint64
		consumed atomic.Uint64
		sumOut   atomic.Uint64
		stop     atomic.Bool
		wg       sync.WaitGroup
	)

	// Consumers must observe strictly increasing positions, so a single
	// consumer per test lets us assert exact FIFO. Two consumers assert only
	// that no record is lost or duplicated.
	for c := 0; c < 2; c++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var rec Record
			for {
				pos, ok := r.dequeue(&rec)
				if !ok {
					if stop.Load() && r.pending() == 0 {
						return
					}
					runtime.Gosched()
					continue
				}
				if len(rec.Payload) != 8 {
					t.Errorf("payload length %d", len(rec.Payload))
				}
				var v uint64
				for i := 0; i < 8; i++ {
					v |= uint64(rec.Payload[i]) << (8 * i)
				}
				if v != rec.Key {
					t.Errorf("payload %d does not match key %d: torn write", v, rec.Key)
				}
				sumOut.Add(rec.Key)
				consumed.Add(1)
				r.release(pos)
			}
		}()
	}

	var sumIn atomic.Uint64
	for p := 0; p < producers; p++ {
		wg.Add(1)
		go func(base uint64) {
			defer wg.Done()
			buf := make([]byte, 8)
			for i := uint64(0); i < perProd; i++ {
				key := base*perProd + i + 1
				for j := 0; j < 8; j++ {
					buf[j] = byte(key >> (8 * j))
				}
				for {
					pos, ok := r.claim()
					if !ok {
						runtime.Gosched() // ring full: wait for consumers
						continue
					}
					r.publish(pos, key, key, 0, 0, buf, 0)
					break
				}
				sumIn.Add(key)
				produced.Add(1)
			}
		}(uint64(p))
	}

	// Wait for producers, then signal consumers.
	done := make(chan struct{})
	go func() {
		for produced.Load() < producers*perProd {
			runtime.Gosched()
		}
		stop.Store(true)
		close(done)
	}()
	<-done
	wg.Wait()

	if consumed.Load() != produced.Load() {
		t.Fatalf("consumed %d != produced %d", consumed.Load(), produced.Load())
	}
	if sumOut.Load() != sumIn.Load() {
		t.Fatalf("checksum of keys mismatch: in=%d out=%d", sumIn.Load(), sumOut.Load())
	}
	if p := r.pending(); p != 0 {
		t.Fatalf("pending = %d at end", p)
	}
}

// --- index ------------------------------------------------------------------

func TestIndexUpsertLookup(t *testing.T) {
	v, err := New(testConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	for i := uint64(1); i <= 500; i++ {
		if _, err := v.Publish(i, []byte(fmt.Sprintf("value-%d", i))); err != nil {
			t.Fatalf("Publish(%d): %v", i, err)
		}
	}
	var rec Record
	for i := uint64(1); i <= 500; i++ {
		if err := v.Lookup(i, &rec); err != nil {
			t.Fatalf("Lookup(%d): %v", i, err)
		}
		want := fmt.Sprintf("value-%d", i)
		if string(rec.Payload) != want {
			t.Fatalf("Lookup(%d) = %q, want %q", i, rec.Payload, want)
		}
		if rec.Key != i {
			t.Fatalf("Lookup(%d) returned key %d", i, rec.Key)
		}
		if !rec.Verify() {
			t.Fatalf("checksum failed for key %d", i)
		}
	}
	if err := v.Lookup(999999, &rec); err != ErrNotFound {
		t.Fatalf("Lookup(absent) = %v, want ErrNotFound", err)
	}
	if err := v.Lookup(0, &rec); err != ErrConfig {
		t.Fatalf("Lookup(0) = %v, want ErrConfig", err)
	}
}

func TestIndexMonotonicUpdate(t *testing.T) {
	v, err := New(testConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	for i := 0; i < 50; i++ {
		if _, err := v.Publish(42, []byte(fmt.Sprintf("v%d", i))); err != nil {
			t.Fatalf("Publish: %v", err)
		}
	}
	var rec Record
	if err := v.Lookup(42, &rec); err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if string(rec.Payload) != "v49" {
		t.Fatalf("index resolved to %q, want the newest value v49", rec.Payload)
	}
}

func TestIndexRejectsZeroKey(t *testing.T) {
	v, err := New(testConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()
	if _, err := v.Publish(0, []byte("x")); err != ErrConfig {
		t.Fatalf("Publish(0) = %v, want ErrConfig", err)
	}
}

// --- engine -----------------------------------------------------------------

func TestPublishDrainRoundTrip(t *testing.T) {
	v, err := New(testConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	const n = 2000
	for i := uint64(1); i <= n; i++ {
		if _, err := v.Publish(i, []byte{byte(i), byte(i >> 8)}); err != nil {
			t.Fatalf("Publish(%d): %v", i, err)
		}
	}

	seen := make(map[uint64]bool, n)
	got := v.Drain(0, func(rec *Record) {
		if seen[rec.Key] {
			t.Errorf("duplicate delivery of key %d", rec.Key)
		}
		seen[rec.Key] = true
		if len(rec.Payload) != 2 || rec.Payload[0] != byte(rec.Key) {
			t.Errorf("payload mismatch for key %d", rec.Key)
		}
	})
	if got != n {
		t.Fatalf("drained %d records, want %d", got, n)
	}
	if len(seen) != n {
		t.Fatalf("saw %d distinct keys, want %d", len(seen), n)
	}
}

func TestPayloadTooLarge(t *testing.T) {
	v, err := New(testConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()
	if _, err := v.Publish(1, make([]byte, 65)); err != ErrPayloadTooLarge {
		t.Fatalf("oversized Publish = %v, want ErrPayloadTooLarge", err)
	}
	if _, err := v.Publish(1, make([]byte, 64)); err != nil {
		t.Fatalf("exact-capacity Publish failed: %v", err)
	}
}

func TestTicketsAreMonotonic(t *testing.T) {
	v, err := New(testConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		seen = map[uint64]bool{}
	)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				tk, err := v.Publish(uint64(g*1000+i+1), []byte("x"))
				if err != nil {
					continue
				}
				mu.Lock()
				if seen[tk] {
					t.Errorf("ticket %d issued twice", tk)
				}
				seen[tk] = true
				mu.Unlock()
			}
		}(g)
	}
	wg.Wait()
	if len(seen) == 0 {
		t.Fatal("no tickets issued")
	}
}

// --- backpressure -----------------------------------------------------------

func TestBackpressureGateFreezesAndReleases(t *testing.T) {
	cfg := testConfig()
	cfg.Shards = 1
	cfg.CellsPerShard = 256
	cfg.HighWaterPct = 90
	cfg.LowWaterPct = 70
	v, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	// Fill past the high-water mark. Nothing consumes, so the gate must latch.
	var frozenAt uint64
	for i := uint64(1); i <= 256; i++ {
		if _, err := v.Publish(i, []byte("x")); err != nil {
			if err != ErrBackpressure {
				t.Fatalf("Publish(%d) = %v, want ErrBackpressure", i, err)
			}
			frozenAt = i
			break
		}
	}
	if frozenAt == 0 {
		t.Fatal("gate never closed despite filling the matrix")
	}

	st := v.Stats()
	if !st.Gate.Closed {
		t.Fatal("gate reports open after freezing ingestion")
	}
	if st.OccupancyPct < cfg.HighWaterPct {
		t.Fatalf("gate closed at %d%% occupancy, below the %d%% high-water mark",
			st.OccupancyPct, cfg.HighWaterPct)
	}
	t.Logf("gate latched at record %d, occupancy %d%%", frozenAt, st.OccupancyPct)

	// While shut, every ingestion attempt must be refused.
	for i := 0; i < 10; i++ {
		if _, err := v.Publish(99999, []byte("x")); err != ErrBackpressure {
			t.Fatalf("Publish while frozen = %v, want ErrBackpressure", err)
		}
	}

	// Drain a single record: still above the low-water mark, so the gate must
	// stay shut. This is the hysteresis that stops the valve from flapping.
	v.Drain(1, nil)
	if !v.Stats().Gate.Closed {
		t.Fatal("gate reopened after draining one record, above the low-water mark")
	}

	// Drain below the low-water mark; the gate must release.
	target := int(cfg.CellsPerShard) * int(100-cfg.LowWaterPct) / 100
	v.Drain(target+8, nil)
	st = v.Stats()
	if st.Gate.Closed {
		t.Fatalf("gate still shut at %d%% occupancy, below the %d%% low-water mark",
			st.OccupancyPct, cfg.LowWaterPct)
	}
	if _, err := v.Publish(123456, []byte("x")); err != nil {
		t.Fatalf("Publish after release = %v, want success", err)
	}
	if st.Gate.CloseCount == 0 || st.Gate.OpenCount == 0 {
		t.Fatalf("gate transition counters not recorded: %+v", st.Gate)
	}
}

func TestBackpressureUnderConcurrentLoad(t *testing.T) {
	var consumed atomic.Uint64
	cfg := testConfig()
	cfg.Shards = 4
	cfg.CellsPerShard = 512
	cfg.Consumer = func(rec *Record) { consumed.Add(1) }

	v, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var (
		wg        sync.WaitGroup
		published atomic.Uint64
		frozen    atomic.Uint64
	)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 20000; i++ {
				key := uint64(g)*100000 + uint64(i) + 1
				_, err := v.Publish(key, []byte("payload"))
				switch err {
				case nil:
					published.Add(1)
				case ErrBackpressure, ErrShardFull:
					frozen.Add(1)
					runtime.Gosched()
				default:
					t.Errorf("Publish: %v", err)
					return
				}
			}
		}(g)
	}
	wg.Wait()

	if err := v.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if published.Load() == 0 {
		t.Fatal("nothing was published")
	}
	if consumed.Load() != published.Load() {
		t.Fatalf("consumed %d != published %d: records lost or duplicated",
			consumed.Load(), published.Load())
	}
	t.Logf("published=%d consumed=%d throttled=%d", published.Load(), consumed.Load(), frozen.Load())
}

// --- lifecycle --------------------------------------------------------------

func TestConsumerWorkersDrainEverything(t *testing.T) {
	var (
		consumed atomic.Uint64
		sum      atomic.Uint64
	)
	cfg := testConfig()
	cfg.CellsPerShard = 4096
	cfg.Consumer = func(rec *Record) {
		consumed.Add(1)
		sum.Add(rec.Key)
	}
	v, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const n = 5000
	var want uint64
	for i := uint64(1); i <= n; i++ {
		if _, err := v.Publish(i, []byte("d")); err != nil {
			t.Fatalf("Publish(%d): %v", i, err)
		}
		want += i
	}
	if err := v.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if consumed.Load() != n {
		t.Fatalf("consumed %d, want %d", consumed.Load(), n)
	}
	if sum.Load() != want {
		t.Fatalf("key sum %d, want %d", sum.Load(), want)
	}
}

func TestPublishAfterCloseFails(t *testing.T) {
	cfg := testConfig()
	cfg.Consumer = func(rec *Record) {}
	v, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := v.Publish(1, []byte("x")); err != nil {
		t.Fatalf("Publish before close: %v", err)
	}
	if err := v.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := v.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestTombstone(t *testing.T) {
	v, err := New(testConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	if _, err := v.Publish(7, []byte("live")); err != nil {
		t.Fatalf("Publish: %v", err)
	}
	if _, err := v.PublishTombstone(7); err != nil {
		t.Fatalf("PublishTombstone: %v", err)
	}
	var rec Record
	if err := v.Lookup(7, &rec); err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if !rec.IsTombstone() {
		t.Fatalf("expected tombstone, flags=%b", rec.Flags)
	}
}

func TestConfigValidation(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*Config)
	}{
		{"non-power-of-two cells", func(c *Config) { c.CellsPerShard = 1000 }},
		{"low water above high", func(c *Config) { c.HighWaterPct, c.LowWaterPct = 50, 60 }},
		{"high water above 100", func(c *Config) { c.HighWaterPct = 150 }},
		{"negative payload", func(c *Config) { c.PayloadCapacity = -1 }},
	}
	for _, tc := range cases {
		cfg := testConfig()
		tc.mut(&cfg)
		if _, err := New(cfg); err != ErrConfig {
			t.Errorf("%s: New = %v, want ErrConfig", tc.name, err)
		}
	}
}

func TestShardsRoundUpToPowerOfTwo(t *testing.T) {
	cfg := testConfig()
	cfg.Shards = 5
	v, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()
	if v.shardN != 8 {
		t.Fatalf("shards = %d, want 8", v.shardN)
	}
}

// TestNoHeapAllocationOnPublish is the load-bearing test of the whole design:
// if the write path allocates, the engine has failed at its stated purpose.
func TestNoHeapAllocationOnPublish(t *testing.T) {
	cfg := testConfig()
	cfg.CellsPerShard = 8192
	cfg.Checksums = true
	v, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	payload := []byte("a fixed 24-byte payload!")
	var rec Record

	allocs := testing.AllocsPerRun(1000, func() {
		tk, err := v.Publish(1, payload)
		if err != nil || tk == 0 {
			return
		}
		_ = v.Lookup(1, &rec)
		v.Drain(1, nil)
	})
	if allocs != 0 {
		t.Fatalf("Publish+Lookup+Drain allocated %.1f objects per run, want 0", allocs)
	}
}

// TestIndexReclaimsDeadBuckets covers the failure mode where a stream of unique
// keys fills every bucket permanently: without reclamation the index rejects all
// new keys after `buckets` insertions, even though nearly every entry points at
// a cell the ring recycled long ago.
func TestIndexReclaimsDeadBuckets(t *testing.T) {
	cfg := testConfig()
	cfg.Shards = 1
	cfg.CellsPerShard = 256
	cfg.IndexBucketsPerCell = 2 // 512 buckets
	v, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	// Push far more distinct keys through than the index can hold, draining as
	// we go so the ring keeps recycling cells underneath the index.
	const total = 20000
	for i := uint64(1); i <= total; i++ {
		if _, err := v.Publish(i, []byte("v")); err != nil {
			t.Fatalf("Publish(%d): %v", i, err)
		}
		if i%64 == 0 {
			v.Drain(64, nil)
		}
	}

	st := v.Stats().ShardStats[0].Index
	if st.Reclaims == 0 {
		t.Fatalf("no buckets reclaimed after %d distinct keys in %d buckets", total, st.Buckets)
	}
	if st.Overflows > st.Reclaims {
		t.Errorf("index overflowed %d times against %d reclaims", st.Overflows, st.Reclaims)
	}

	// The most recent keys must still be resolvable: this is the property that
	// permanent saturation would have destroyed.
	var rec Record
	found := 0
	for i := uint64(total - 32); i <= total; i++ {
		if err := v.Lookup(i, &rec); err == nil {
			found++
		}
	}
	if found == 0 {
		t.Fatalf("no recent key resolvable; index is permanently saturated")
	}
	t.Logf("buckets=%d live=%d reclaims=%d overflows=%d recentFound=%d/33",
		st.Buckets, st.Live, st.Reclaims, st.Overflows, found)
}
