// Command vaultagectl exercises the Vaultage engine and reports the physical
// properties of its off-heap memory spine.
//
// It exists to make the engine's guarantees observable rather than merely
// documented: it prints the actual virtual addresses the kernel returned, proves
// they sit on 2MiB boundaries, reports whether the pages were pinned, and shows
// the backpressure valve latching and releasing under a deliberately
// overdriven load.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vaultage/vaultage"
)

func main() {
	var (
		shards    = flag.Int("shards", runtime.GOMAXPROCS(0), "number of shards (rounded up to a power of two)")
		cells     = flag.Uint64("cells", 1<<16, "cells per shard (power of two)")
		payload   = flag.Int("payload", 448, "payload bytes per cell")
		producers = flag.Int("producers", runtime.GOMAXPROCS(0), "concurrent producer goroutines")
		records   = flag.Int("records", 2_000_000, "records per producer")
		duration  = flag.Duration("duration", 0, "stop after this long (0 = run to completion)")
		huge      = flag.Bool("require-huge", false, "fail unless the kernel grants huge pages")
		pinned    = flag.Bool("require-pinned", false, "fail unless mlock succeeds")
		coarse    = flag.Duration("coarse-clock", 0, "cache the clock at this period (0 = precise time.Now per record)")
		slow      = flag.Bool("slow-consumer", false, "throttle the consumer to force backpressure")
	)
	flag.Parse()

	cfg := vaultage.DefaultConfig()
	cfg.Shards = *shards
	cfg.CellsPerShard = *cells
	cfg.PayloadCapacity = *payload
	cfg.RequireHugePages = *huge
	cfg.RequirePinned = *pinned
	cfg.CoarseClock = *coarse

	var (
		consumed atomic.Uint64
		bytes    atomic.Uint64
	)
	cfg.Consumer = func(rec *vaultage.Record) {
		consumed.Add(1)
		bytes.Add(uint64(len(rec.Payload)))
		if *slow {
			// Deliberately fall behind so the high-water mark is crossed.
			time.Sleep(time.Microsecond)
		}
	}

	eng, err := vaultage.New(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "vaultage: %v\n", err)
		os.Exit(1)
	}

	printArenaReport(eng)

	// ---- drive load ------------------------------------------------------
	var (
		wg          sync.WaitGroup
		published   atomic.Uint64
		backpressed atomic.Uint64
		full        atomic.Uint64
		deadline    time.Time
	)
	if *duration > 0 {
		deadline = time.Now().Add(*duration)
	}

	start := time.Now()
	for p := 0; p < *producers; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			buf := make([]byte, *payload)
			for i := 0; i < *records; i++ {
				if !deadline.IsZero() && i%1024 == 0 && time.Now().After(deadline) {
					return
				}
				key := uint64(p)*uint64(*records) + uint64(i) + 1
				switch _, err := eng.Publish(key, buf); err {
				case nil:
					published.Add(1)
				case vaultage.ErrBackpressure:
					backpressed.Add(1)
					runtime.Gosched()
				case vaultage.ErrShardFull:
					full.Add(1)
					runtime.Gosched()
				default:
					fmt.Fprintf(os.Stderr, "publish: %v\n", err)
					return
				}
			}
		}(p)
	}
	wg.Wait()
	elapsed := time.Since(start)

	st := eng.Stats()
	if err := eng.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "close: %v\n", err)
	}

	printThroughput(elapsed, published.Load(), consumed.Load(), backpressed.Load(), full.Load(), bytes.Load())
	printGateReport(st)
	printShardReport(st)
}

func printArenaReport(eng *vaultage.Vaultage) {
	st := eng.Stats()
	fmt.Println("=== OFF-HEAP MEMORY SPINE ===")
	fmt.Printf("  shards              : %d\n", st.Shards)
	fmt.Printf("  cell stride         : %d bytes (header %d + payload %d)\n",
		st.CellStride, 64, st.PayloadCap)
	fmt.Printf("  mapped off-heap     : %.1f MiB\n", float64(st.ArenaBytes)/(1<<20))
	fmt.Printf("  pinned (mlock)      : %.1f MiB\n", float64(st.PinnedBytes)/(1<<20))
	fmt.Printf("  2MiB aligned        : %v\n", st.HugeAligned)
	fmt.Printf("  huge pages (native) : %v\n", st.HugeNative)
	fmt.Println()
	fmt.Println("  arena base addresses (all must end in 0x00000 to be 2MiB aligned):")
	for _, ss := range st.ShardStats {
		a := ss.Arena
		fmt.Printf("    shard %2d: %#016x  size=%7d KiB  page=%5d  aligned=%v pinned=%v\n",
			ss.ID, a.Base, a.Size/1024, a.PageSize,
			a.Base%(2<<20) == 0, a.Pinned)
	}
	fmt.Println()
}

func printThroughput(elapsed time.Duration, published, consumed, backpressed, full, bytes uint64) {
	secs := elapsed.Seconds()
	fmt.Println("=== THROUGHPUT ===")
	fmt.Printf("  elapsed             : %v\n", elapsed.Round(time.Millisecond))
	fmt.Printf("  published           : %d (%.2f M/s)\n", published, float64(published)/secs/1e6)
	fmt.Printf("  consumed            : %d (%.2f M/s)\n", consumed, float64(consumed)/secs/1e6)
	fmt.Printf("  payload throughput  : %.2f GiB/s\n", float64(bytes)/secs/(1<<30))
	fmt.Printf("  ns per record       : %.1f\n", float64(elapsed.Nanoseconds())/float64(max64(published, 1)))
	fmt.Printf("  refused (gate)      : %d\n", backpressed)
	fmt.Printf("  refused (shard full): %d\n", full)
	fmt.Println()

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Println("=== GO HEAP (the point: this does not grow with record count) ===")
	fmt.Printf("  heap in use         : %.2f MiB\n", float64(ms.HeapInuse)/(1<<20))
	fmt.Printf("  total allocations   : %d objects\n", ms.Mallocs)
	fmt.Printf("  GC cycles           : %d\n", ms.NumGC)
	fmt.Printf("  total GC pause      : %v\n", time.Duration(ms.PauseTotalNs))
	fmt.Println()
}

func printGateReport(st vaultage.Stats) {
	fmt.Println("=== BACKPRESSURE VALVE ===")
	fmt.Printf("  high / low water    : %d%% / %d%%\n", st.Gate.HighWaterPct, st.Gate.LowWaterPct)
	fmt.Printf("  currently closed    : %v\n", st.Gate.Closed)
	fmt.Printf("  freeze episodes     : %d\n", st.Gate.CloseCount)
	fmt.Printf("  releases            : %d\n", st.Gate.OpenCount)
	fmt.Printf("  ingestion refused   : %d\n", st.Gate.RejectedCount)
	fmt.Println()
}

func printShardReport(st vaultage.Stats) {
	fmt.Println("=== SHARDS ===")
	fmt.Printf("  %-6s %10s %10s %10s %8s %10s %10s\n",
		"shard", "ingested", "consumed", "rejected", "occ%", "commitIdx", "idxLoad%")
	for _, ss := range st.ShardStats {
		fmt.Printf("  %-6d %10d %10d %10d %7d%% %10d %9d%%\n",
			ss.ID, ss.Ingested, ss.Consumed, ss.Rejected,
			ss.OccupancyPct, ss.CommitIndex, ss.Index.LoadPct)
	}
}

func max64(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}
