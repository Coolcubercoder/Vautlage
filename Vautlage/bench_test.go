package vaultage

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func benchEngine(b *testing.B, shards int, cells uint64) *Vaultage {
	b.Helper()
	cfg := DefaultConfig()
	cfg.Shards = shards
	cfg.CellsPerShard = cells
	cfg.PayloadCapacity = 64
	cfg.Checksums = false
	cfg.PinWorkerThreads = false
	cfg.Consumer = nil
	v, err := New(cfg)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	return v
}

// liveEngine builds an engine with real consumer threads draining every shard,
// so the benchmarks below measure the steady-state write path rather than the
// saturation path. Without live consumers the matrix fills, the gate latches,
// and Publish starts returning ErrBackpressure in a couple of nanoseconds --
// which looks spectacular in a benchmark and measures nothing at all.
func liveEngine(b *testing.B, shards int, cells uint64) (*Vaultage, *atomic.Uint64) {
	b.Helper()
	var consumed atomic.Uint64
	cfg := DefaultConfig()
	cfg.Shards = shards
	cfg.CellsPerShard = cells
	cfg.PayloadCapacity = 64
	cfg.Checksums = false
	cfg.PinWorkerThreads = false
	cfg.Consumer = func(rec *Record) { consumed.Add(1) }
	v, err := New(cfg)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	return v, &consumed
}

// reportThrottle records what fraction of publishes were refused, so a headline
// ns/op figure can never quietly be the cost of a rejection.
func reportThrottle(b *testing.B, throttled uint64) {
	b.ReportMetric(100*float64(throttled)/float64(b.N), "%throttled")
}

// BenchmarkPublish measures the single-threaded write path with a live consumer:
// gate check, ticket, claim CAS, memmove, header stores, release store,
// commit-index advance, and the index CAS.
func BenchmarkPublish(b *testing.B) {
	v, _ := liveEngine(b, 1, 1<<16)
	defer v.Close()
	payload := make([]byte, 32)
	var throttled uint64

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := v.Publish(uint64(i)+1, payload); err != nil {
			throttled++
		}
	}
	b.StopTimer()
	reportThrottle(b, throttled)
}

// BenchmarkPublishCoarseClock is the same path with the cached clock, isolating
// the cost of time.Now on the write path.
func BenchmarkPublishCoarseClock(b *testing.B) {
	var consumed atomic.Uint64
	cfg := DefaultConfig()
	cfg.Shards = 1
	cfg.CellsPerShard = 1 << 16
	cfg.PayloadCapacity = 64
	cfg.Checksums = false
	cfg.PinWorkerThreads = false
	cfg.CoarseClock = 100 * time.Microsecond
	cfg.Consumer = func(rec *Record) { consumed.Add(1) }
	v, err := New(cfg)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	defer v.Close()

	payload := make([]byte, 32)
	var throttled uint64
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := v.Publish(uint64(i)+1, payload); err != nil {
			throttled++
		}
	}
	b.StopTimer()
	reportThrottle(b, throttled)
}

// BenchmarkPublishParallel is the scaling benchmark: aggregate throughput as
// producer count rises, with one shard and one consumer thread per core.
func BenchmarkPublishParallel(b *testing.B) {
	v, _ := liveEngine(b, 8, 1<<16)
	defer v.Close()

	var throttled atomic.Uint64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		payload := make([]byte, 32)
		var i uint64
		for pb.Next() {
			i++
			if _, err := v.Publish(i, payload); err != nil {
				throttled.Add(1)
			}
		}
	})
	b.StopTimer()
	reportThrottle(b, throttled.Load())
}

// BenchmarkPublishParallelCoarse is the scaling benchmark with the cached clock.
func BenchmarkPublishParallelCoarse(b *testing.B) {
	var consumed atomic.Uint64
	cfg := DefaultConfig()
	cfg.Shards = 8
	cfg.CellsPerShard = 1 << 16
	cfg.PayloadCapacity = 64
	cfg.Checksums = false
	cfg.PinWorkerThreads = false
	cfg.CoarseClock = 100 * time.Microsecond
	cfg.Consumer = func(rec *Record) { consumed.Add(1) }
	v, err := New(cfg)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	defer v.Close()

	var throttled atomic.Uint64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		payload := make([]byte, 32)
		var i uint64
		for pb.Next() {
			i++
			if _, err := v.Publish(i, payload); err != nil {
				throttled.Add(1)
			}
		}
	})
	b.StopTimer()
	reportThrottle(b, throttled.Load())
}

// BenchmarkRingClaimPublish measures the ring turnstile alone, with no ticket
// dispenser, no index and no clock. This is the floor: the cost of the
// lock-free slot handoff itself.
func BenchmarkRingClaimPublish(b *testing.B) {
	a, err := NewArena(ringBytes(1<<16, 64), false, false)
	if err != nil {
		b.Fatalf("NewArena: %v", err)
	}
	defer a.Close()
	r := new(ring)
	if err := initRing(r, a.At(0), 1<<16, 64, 1); err != nil {
		b.Fatalf("initRing: %v", err)
	}
	payload := make([]byte, 32)
	var rec Record

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pos, ok := r.claim()
		if !ok {
			for {
				dp, ok := r.dequeue(&rec)
				if !ok {
					break
				}
				r.release(dp)
			}
			continue
		}
		r.publish(pos, uint64(i), uint64(i)+1, 0, 0, payload, 0)
	}
}

// BenchmarkLookup measures the read path: hash, probe, cell validation, and a
// zero-copy payload alias.
func BenchmarkLookup(b *testing.B) {
	v, _ := liveEngine(b, 4, 1<<14)
	defer v.Close()
	for i := uint64(1); i <= 1000; i++ {
		v.Publish(i, []byte("value"))
	}
	var rec Record

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v.Lookup(uint64(i%1000)+1, &rec)
	}
}

// BenchmarkHeapBaseline is the thing Vaultage exists to replace: a mutex-guarded
// map of heap-allocated records. It is included so the comparison in the README
// is measured rather than asserted.
func BenchmarkHeapBaseline(b *testing.B) {
	type heapRecord struct {
		Ticket    uint64
		Timestamp int64
		Key       uint64
		Payload   []byte
	}
	var (
		mu     sync.Mutex
		state  = make(map[uint64]*heapRecord, 1<<16)
		ticket uint64
	)
	payload := make([]byte, 32)

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var i uint64
		for pb.Next() {
			i++
			buf := make([]byte, len(payload)) // per-mutation heap allocation
			copy(buf, payload)
			r := &heapRecord{Key: i, Payload: buf, Timestamp: time.Now().UnixNano()}
			mu.Lock()
			ticket++
			r.Ticket = ticket
			state[i&0xFFFF] = r
			mu.Unlock()
		}
	})
}
