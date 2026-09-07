# Vaultage

**An extreme-throughput, memory-bounded state engine for Go. Zero dependencies outside the standard library. Zero heap allocations on the hot path. Zero garbage collector involvement in the storage matrix.**

Vaultage is the core engine of a distributed state fabric. It tracks millions of volatile state mutations per second without ever handing one to the Go heap: the entire storage matrix is mapped off-heap with a raw `mmap` syscall, aligned to 2 MiB huge-page boundaries, pinned into physical RAM with `mlock`, and accessed exclusively through lock-free `CompareAndSwapUint64` turnstiles.

```
1,200,000 records published and consumed
    →  333 total heap allocations
    →  0 garbage collection cycles
    →  0s total GC pause
```

---

## Table of contents

- [The physical problem](#the-physical-problem)
- [Architecture](#architecture)
- [1. Huge-page aligned off-heap memory spine](#1-huge-page-aligned-off-heap-memory-spine)
- [2. Packed, strict 8-byte aligned storage matrix](#2-packed-strict-8-byte-aligned-storage-matrix)
- [3. Inter-core cache isolation](#3-inter-core-cache-isolation)
- [4. Non-blocking atomic ticket handoff](#4-non-blocking-atomic-ticket-handoff)
- [The exact pointer arithmetic](#the-exact-pointer-arithmetic)
- [Backpressure](#backpressure)
- [API](#api)
- [Measured performance](#measured-performance)
- [Platform support](#platform-support)
- [Operational requirements](#operational-requirements)
- [Design trade-offs and limitations](#design-trade-offs-and-limitations)
- [Verification](#verification)
- [Source map](#source-map)

---

## The physical problem

At high mutation rates the bottleneck stops being algorithmic and becomes physical. A conventional Go implementation — allocate a record, put it in a map, let the GC clean up — fails on four distinct pieces of hardware simultaneously.

**The TLB.** A CPU caches virtual→physical page translations in a Translation Lookaside Buffer holding roughly 1,500–3,000 entries. A 1 GiB working set addressed with 4 KiB pages requires **262,144** translations. The TLB can hold under 1% of them, so nearly every access to a cold record triggers a hardware page walk — four dependent memory reads before the actual load even begins. The same 1 GiB on 2 MiB huge pages requires **512** translations, which fits.

| Page size | Translations for 1 GiB | Fits in TLB? |
|---|---:|---|
| 4 KiB | 262,144 | No — thrashes |
| 16 KiB (Apple Silicon) | 65,536 | No — thrashes |
| **2 MiB (huge)** | **512** | **Yes** |

**The cache.** Every heap allocation dirties a fresh 64-byte line, evicting live working-set data. Worse, two hot variables that land in the same line — a producer cursor and a consumer cursor, say — turn a lock-free algorithm into a hardware mutex: each core's write invalidates the other's cached copy, and throughput *falls* as cores are added. This is false sharing, and it is invisible in profiles.

**The collector.** Millions of live heap objects must be traced. Write barriers fire on every pointer store, and mark assists steal exactly the cores doing the work.

**The kernel.** Unpinned pages can be swapped. A 100 ns cache miss becomes a multi-millisecond disk fault, and tail latency ceases to exist as a controllable quantity.

Vaultage answers each one with a specific hardware countermeasure rather than a heuristic.

---

## Architecture

```
                             Producers (any goroutine, any core)
                                          │
                            ┌─────────────┴─────────────┐
                            │   atomic backpressure     │   1 atomic load per publish
                            │   gate  (90% / 70%)       │   closed ⇒ ErrBackpressure
                            └─────────────┬─────────────┘
                                          │
                       hashKey(key) & (shards-1)  ── routes to a shard
                                          │
     ┌────────────────────────────────────┼────────────────────────────────────┐
     │                                    │                                    │
┌────▼──────────────────┐       ┌─────────▼─────────────┐       ┌──────────────▼────────┐
│  SHARD 0              │       │  SHARD 1              │       │  SHARD N              │
│  own 2 MiB arena      │       │  own 2 MiB arena      │       │  own 2 MiB arena      │
│  ┌─────────────────┐  │       │                       │       │                       │
│  │ ring cells      │  │       │        ...            │       │        ...            │
│  │ (mmap, mlock)   │  │       │                       │       │                       │
│  ├─────────────────┤  │       │                       │       │                       │
│  │ key index       │  │       │                       │       │                       │
│  └─────────────────┘  │       │                       │       │                       │
│  tail   ┃ own line    │       │                       │       │                       │
│  commit ┃ own line    │       │                       │       │                       │
│  head   ┃ own line    │       │                       │       │                       │
└────┬──────────────────┘       └─────────┬─────────────┘       └──────────────┬────────┘
     │                                    │                                    │
┌────▼──────────────────┐       ┌─────────▼─────────────┐       ┌──────────────▼────────┐
│ worker 0              │       │ worker 1              │       │ worker N              │
│ LockOSThread          │       │ LockOSThread          │       │ LockOSThread          │
│ pinned to core 0      │       │ pinned to core 1      │       │ pinned to core N      │
└───────────────────────┘       └───────────────────────┘       └───────────────────────┘
```

Nothing in the data plane is shared between shards. Each shard owns a separate 2 MiB-aligned mapping, its cursors sit on separate cache lines, and its consumer runs on a separate thread bound to a separate core.

---

## 1. Huge-page aligned off-heap memory spine

The arena is obtained with a direct syscall — not `syscall.Mmap`, whose wrapper maintains its own bookkeeping and returns a Go slice, imposing exactly the runtime coupling this engine exists to avoid.

```go
r, _, errno := syscall.Syscall6(
    syscall.SYS_MMAP,
    0,                                            // addr: kernel chooses
    length,                                       // length
    syscall.PROT_READ|syscall.PROT_WRITE,
    syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS,
    ^uintptr(0),                                  // fd = -1
    0,                                            // offset
)
```

### Forcing 2 MiB alignment

**The kernel does not return aligned addresses.** Measured on the development host, a plain anonymous `mmap` returned `0x10bcb0000` — not 2 MiB aligned. Alignment must be *forced*, and Vaultage does it in two tiers:

**Tier 1 — explicit huge pages (Linux).** Request `MAP_HUGETLB | MAP_HUGE_2MB`. A mapping served from the reserved huge-page pool is 2 MiB aligned by construction. This is the only tier that *guarantees* huge pages rather than merely making them likely; it fails with `ENOMEM` unless the operator reserved pages.

**Tier 2 — over-allocate and trim (portable).** Map `size + 2 MiB`, round the returned address up to the next 2 MiB boundary, then `munmap` the slack on both sides:

```
   raw               aligned base            aligned end          raw end
    │<-- head slack -->│<------- size ------->│<-- tail slack -->│
    │                  │                      │                  │
    ├──────────────────┼──────────────────────┼──────────────────┤
         munmap'd              kept                  munmap'd
```

Over-allocating by exactly one huge page is sufficient: for any returned address, the next 2 MiB boundary is less than 2 MiB away, so the aligned window always fits. The slack is returned to the OS immediately, so the process pays no RSS for it. `madvise(MADV_HUGEPAGE)` then requests transparent huge pages over the aligned window, and `MADV_DONTFORK` prevents a `fork()` from marking the entire matrix copy-on-write.

Proof it works — every base ends in `0x00000`:

```
shard  0: 0x000000012c200000  size=2048 KiB  page=16384  aligned=true pinned=true
shard  1: 0x000000012c400000  size=2048 KiB  page=16384  aligned=true pinned=true
shard  2: 0x000000012c600000  size=2048 KiB  page=16384  aligned=true pinned=true
shard  3: 0x000000012c800000  size=2048 KiB  page=16384  aligned=true pinned=true
```

### Pinning

Every page is prefaulted with a real store (a *read* of anonymous memory can be satisfied by the shared zero page, which allocates no frame and leaves a copy-on-write fault waiting on the hot path), then locked:

```go
func sysPin(p unsafe.Pointer, length uintptr) error {
    return syscall.Mlock(unsafe.Slice((*byte)(p), length))
}
```

With `Config.RequirePinned`, construction *fails* rather than silently running a swappable arena.

---

## 2. Packed, strict 8-byte aligned storage matrix

The span is sliced into fixed-stride cells. The header is exactly one cache line, so touching a record's metadata costs precisely one line fill and never straddles two.

```
 byte 0                      byte 64                    byte 64+payloadCap
 ┌───────────────────────────┬──────────────────────────┐
 │    cellHeader (64 B)      │        payload           │ ...pad to 64 B...
 └───────────────────────────┴──────────────────────────┘
 │<─── exactly one cache line ───>│
```

| Offset | Width | Field | Purpose |
|---:|---:|---|---|
| 0 | 8 | `seq` | turnstile sequence — the lock-free slot state |
| 8 | 8 | `ticket` | globally ordered id assigned at checkout |
| 16 | 8 | `timestamp` | commit time, Unix nanoseconds |
| 24 | 8 | `key` | caller state key |
| 32 | 8 | `length` | payload length |
| 40 | 8 | `epoch` | arena generation |
| 48 | 4 | `checksum` | FNV-1a payload digest |
| 52 | 4 | `flags` | committed / tombstone / truncated |
| 56 | 8 | `reserved` | pads the header to a full line |

`seq` sits at offset 0 deliberately: the turnstile CAS is the hottest instruction in the engine, and offset 0 makes the atomic's address *identical* to the cell's base address — one address computation, no displacement arithmetic in the inner loop.

### Compile-time offset assertions

Offsets are not merely tested, they are **proven at build time**. Converting a negative constant to `uint` is a compile error, so asserting the difference in *both* directions pins each offset to exactly one legal value:

```go
const (
    _ = uint(unsafe.Offsetof(cellHeader{}.ticket) - offTicket)
    _ = uint(offTicket - unsafe.Offsetof(cellHeader{}.ticket))
)
```

If a field is ever reordered, resized, or repadded, **the package does not compile**. The same technique asserts that the header is exactly 64 bytes, that every 64-bit field carries 8-byte alignment, that `padU64` is exactly one coherence stride, that the ring's three cursors are exactly one stride apart, and that the build is 64-bit (a 32-bit build would give `uint64` 4-byte alignment and silently break every atomic in the package).

This matters physically: a lock instruction that straddles two cache lines degrades from a cache-local operation into a bus-wide **split lock**, costing on the order of 100× and stalling every other core on the socket. In Vaultage an unaligned atomic is structurally unreachable, not merely untested — the arena base is 2 MiB aligned, the cell stride is a multiple of 64, and every field offset is a multiple of 8.

---

## 3. Inter-core cache isolation

```go
type padU64 struct {
    v uint64
    _ [CoherenceStride - WordSize]byte
}
```

**The stride is 128 bytes, not 64.** A single 64-byte line defeats classical false sharing, but the adjacent-line prefetchers on both x86-64 (spatial prefetcher, 128-byte sector pairs) and Apple Silicon speculatively pull the sibling line into the same L2 sector. Two writers on sibling lines therefore still ping-pong. A 128-byte stride puts every hot cursor in its own sector pair.

Isolated this way: `tail`, `commit`, `head` per shard; the global ticket dispenser, running flag, epoch, drop counter, saturation counter and coarse clock; every statistics counter; and the gate's state word.

### Why the control plane is also off-heap

A Go heap allocation is only guaranteed **8-byte** alignment. If the shard structs sat on the heap, their padding would be *relative* padding with no fixed relationship to real cache lines, and two shards' cursors could still share a line depending on the size class the allocator happened to pick. Vaultage places the shard control blocks in their own 2 MiB-aligned arena at a 128-byte-multiple stride, making the isolation **exact and deterministic instead of probable**.

This is legal because neither `shard` nor `globalCtl` contains a single Go pointer — every field is an integer, an `unsafe.Pointer` into off-heap memory, or padding. The collector has nothing to trace.

### Thread affinity

Padding stops two cores from sharing a line. Affinity stops one worker from migrating between cores and dragging its entire working set through a cold L1/L2 on the destination. Each worker calls `runtime.LockOSThread` (without it the scheduler could move the goroutine to another thread and undo the affinity on every preemption) and then binds that thread with `sched_setaffinity(2)`:

```go
const setWords = 1024 / 64
var set [setWords]uint64
set[core/64] |= 1 << (uint(core) % 64)
syscall.Syscall(syscall.SYS_SCHED_SETAFFINITY, 0, unsafe.Sizeof(set), uintptr(unsafe.Pointer(&set[0])))
```

---

## 4. Non-blocking atomic ticket handoff

Each cell's `seq` word encodes its state **as a position rather than a flag**. For a ring of capacity `C` at position `pos`:

| `seq` value | Meaning |
|---|---|
| `pos` | free; belongs to the producer at position `pos` |
| `pos + 1` | published; belongs to a consumer |
| `pos + C` | consumed; free for the *next* lap's producer |

Encoding state as a monotonically increasing position makes the ring **ABA-immune without tag words, hazard pointers or epoch reclamation**: a stale CAS from a descheduled thread carries an old position that can never match again, because `seq` only moves forward by one lap at a time.

### The write turnstile

```go
func (r *ring) claim() (pos uint64, ok bool) {
    for {
        pos = r.tail.load()
        c := r.cell(pos)
        seq := ptrLoad64(fieldAt(c, offSeq))

        dif := int64(seq) - int64(pos)   // signed: correct across uint64 wrap

        switch {
        case dif == 0:
            if r.tail.cas(pos, pos+1) {
                return pos, true          // ticket acquired
            }
        case dif < 0:
            return 0, false               // ring saturated
        default:
            // stale cursor read; reload and retry
        }
    }
}
```

The signed difference *is* the algorithm. `dif == 0` means the cell is free and it is our turn. `dif < 0` means the cell still holds an unconsumed record from the previous lap — the ring is full, so report saturation rather than spin. `dif > 0` means another producer won the race and our cursor view is stale.

### Publication ordering

Every header field and payload byte is written with a plain store; only then is `seq` published with an atomic store. In the Go memory model that store is sequentially consistent, so **any core observing `seq == pos+1` is guaranteed to observe every preceding write**. No fence intrinsic is required, and none exists in portable Go.

### The commit index

Producers finish out of order — the producer holding position 7 may publish before the one holding position 5. `commit` is the **contiguous** committed prefix, advancing only when the slot it points at has actually been published:

```
positions:   0    1    2    3    4    5    6    7
published:   ✓    ✓    ✓    ✗    ✓    ✓    ✗    ✓
                            │
             commit ────────┘   (stops here; 4,5,7 are written but not yet visible)
```

Consumers are gated on `commit`, never on `tail`, so a consumer can **never** observe a slot a producer has claimed but not finished writing — the gate is a data dependency, not a lock. Records are therefore delivered in exact ticket order, which is what makes the stream replicable.

Every producer runs the advance loop, so no single thread owns the job. If the producer that would have advanced the index is descheduled mid-loop, the next producer through completes the work; the index is never left behind by a sleeping thread.

### Slot release

```go
func (r *ring) release(pos uint64) {
    ptrStore64(fieldAt(r.cell(pos), offSeq), pos+r.capacity)
}
```

`pos + capacity` is precisely the position of the producer who will own this cell on the next lap. That single store is the entire free-list: there is no free list, no bitmap and no allocator, just an arithmetic identity.

---

## The exact pointer arithmetic

Every address in the engine is a composition of these operations. There is no slice indexing, no bounds check, and no intermediate Go object — the compiler emits a shift, an add, and the memory instruction.

```go
// Arena origin: base + off
func (a *Arena) At(off uintptr) unsafe.Pointer { return unsafe.Add(a.base, off) }

// Cell address: base + idx*stride
func cellAt(base unsafe.Pointer, idx, stride uintptr) unsafe.Pointer {
    return unsafe.Add(base, idx*stride)
}

// Header field: cellBase + fieldOffset
func fieldAt(cellBase unsafe.Pointer, off uintptr) unsafe.Pointer {
    return unsafe.Add(cellBase, off)
}

// Payload region: cellBase + 64
func payloadAt(cellBase unsafe.Pointer) unsafe.Pointer {
    return unsafe.Add(cellBase, HeaderSize)
}

// Ring position → cell: base + (pos & mask) * stride
func (r *ring) cell(pos uint64) unsafe.Pointer {
    return cellAt(r.base, uintptr(pos&r.mask), r.stride)
}

// Index bucket: base + i*32
func (x *keyIndex) bucket(i uint64) unsafe.Pointer {
    return unsafe.Add(x.base, uintptr(i)*IndexBucketSize)
}

// Shard control block: shardBase + i*shardStride
func (v *Vaultage) shard(i int) *shard {
    return (*shard)(unsafe.Add(v.shardPtr, uintptr(i)*shardStride))
}

// Zero-copy payload: a Go slice header pointing straight into mmap'd memory
func payloadSlice(cellBase unsafe.Pointer, n uintptr) []byte {
    return unsafe.Slice((*byte)(payloadAt(cellBase)), n)
}
```

Address arithmetic is carried in `unsafe.Pointer` and advanced with `unsafe.Add`, never round-tripped through `uintptr`. Off-heap addresses are immovable, so a `uintptr` round trip would in fact be safe here, but the pointer form is provably correct to `go vet -unsafeptr` and immune to future changes in how the runtime treats derived addresses.

Shard arena layout — one flat mapping, two regions:

```
+0                    ring cells        (cells × stride)
+ringBytes            index buckets     (buckets × 32)
```

`ringBytes` is a multiple of the cell stride, itself a multiple of 64, so the index base inherits cache-line alignment from the 2 MiB-aligned arena base with no explicit padding.

---

## Backpressure

A single atomic word gates all ingestion. Every `Publish` begins with one uncontended load of a line that is almost always shared-clean.

**Hysteresis is mandatory.** A single threshold produces flapping: occupancy hovers at the limit, the gate opens and closes on alternate records, and every producer pays a contended CAS on the same line for every operation — precisely the false-sharing storm the rest of the engine exists to prevent. With a **90% high-water** and **70% low-water** mark, the gate closes once and stays closed until consumers have reclaimed a fifth of the matrix, so the CAS fires roughly twice per saturation episode instead of twice per record.

**The threshold test is O(1), not O(shards).** An earlier revision swept every shard's cursors on the write path; that made every producer pull every *other* shard's cache lines on every record, and measured throughput *fell* as cores were added. The shipped design latches a per-shard `hot` flag on its own cache line, and only the single producer that observes the **transition** touches the global counter and the gate.

Freezing engine-wide on one hot shard is deliberate: keys route to shards by hash, so a caller cannot steer writes away from a saturated shard. An engine that kept accepting writes because *other* shards had room would be bounded only by luck.

Observed under a deliberately throttled consumer:

```
=== BACKPRESSURE VALVE ===
  high / low water    : 90% / 70%
  freeze episodes     : 6
  releases            : 5
  ingestion refused   : 155622

  refused (gate)      : 155622
  refused (shard full): 0        ← the gate caught saturation before the
                                    ring could ever overflow
```

Zero shard-full rejections is the result that matters: backpressure engaged, rather than data being lost at the boundary.

---

## API

```go
cfg := vaultage.DefaultConfig()
cfg.Shards          = 8
cfg.CellsPerShard   = 1 << 16   // power of two
cfg.PayloadCapacity = 448       // → exact 512-byte cell stride
cfg.RequirePinned   = true      // fail if mlock is unavailable
cfg.Consumer = func(rec *vaultage.Record) {
    // rec.Payload aliases off-heap memory and is valid ONLY until this
    // callback returns. Call rec.CopyPayload() to retain it.
    process(rec.Key, rec.Payload)
}

eng, err := vaultage.New(cfg)
if err != nil {
    return err
}
defer eng.Close()

ticket, err := eng.Publish(key, payload)
switch err {
case nil:
case vaultage.ErrBackpressure:  // engine frozen; retry after draining
case vaultage.ErrShardFull:     // this shard saturated
case vaultage.ErrPayloadTooLarge:
}

var rec vaultage.Record
if err := eng.Lookup(key, &rec); err == nil {
    use(rec.Payload)            // zero-copy, zero-allocation
}
```

| Method | Purpose |
|---|---|
| `New(Config)` | maps, pins and lays out every arena; starts workers |
| `Publish(key, payload)` | ingest one mutation, returns its ticket |
| `PublishTombstone(key)` | record a logical deletion |
| `Lookup(key, *Record)` | resolve a key to its newest record, zero-copy |
| `Drain(max, Handler)` | synchronous consumption when no `Consumer` is set |
| `Stats()` | arena addresses, occupancy, gate and index health |
| `Close()` | stop, drain, join workers, unpin, unmap |

Run the demo:

```bash
go run ./cmd/vaultagectl -shards 4 -cells 8192 -producers 4 -records 300000
go run ./cmd/vaultagectl -slow-consumer          # watch the gate latch and release
go run ./cmd/vaultagectl -require-huge -require-pinned   # enforce hardware guarantees
```

---

## Measured performance

Apple M4 (10 cores), darwin/arm64, Go 1.27, `-benchtime=500ms -cpu=8 -count=3`.

| Benchmark | ns/op | B/op | allocs/op |
|---|---:|---:|---:|
| `Publish` (end-to-end, live consumer) | 91.0 | 0 | **0** |
| `Publish` (coarse clock) | 72.7 | 0 | **0** |
| `PublishParallel` (8 producers) | 85.8 | 0 | **0** |
| `RingClaimPublish` (turnstile alone) | 11.6 | 0 | **0** |
| `Lookup` (zero-copy) | 8.8 | 0 | **0** |
| `HeapBaseline` (mutex + map + heap records) | 147.2 | 80 | **2** |

**Read these numbers with the following caveats.**

- **The `%throttled` metric exists because benchmarks lie.** An early run showed `PublishConsume` at 2.15 ns/op — that was not throughput, it was `Publish` returning `ErrBackpressure` in two nanoseconds because the consumers never got scheduled. Every publish benchmark now reports the fraction of calls that were refused; only rows at ~0% measure real work.
- **`time.Now()` is 27 ns of the 91 ns write path** — over half. Setting `Config.CoarseClock` swaps it for a cached reading refreshed by a background ticker, cutting publish to ~73 ns. Record *ordering* never depends on the clock (that is what the ticket is for), so this costs only timestamp granularity. It is off by default: correctness first, throughput opt-in.
- **Run-to-run variance across invocations is ~20%** on this laptop (thermal, and P-core vs E-core placement). Differences under ~15 ns here are noise. The trustworthy comparisons are the large ones: allocation counts, and the *direction* each design moves as cores are added.
- **This host has no true huge pages.** darwin/arm64 rejects the superpage flag, so the arena falls back to a 2 MiB-aligned mapping of 16 KiB pages. The TLB benefits of alignment are partially realized; the full benefit requires Linux with reserved huge pages.
- **The parallel benchmark is pipeline-bound, not producer-bound.** Eight producer goroutines plus eight spinning consumer threads oversubscribe a 10-core machine, so it does **not** demonstrate clean linear scaling. What it does show is that the number stays flat rather than degrading — contrast `HeapBaseline`, which goes 62 ns → 148 ns from 1 to 8 cores under mutex and allocator contention.

The clearest result is not a latency figure at all:

```
1.2M records:  333 total heap allocations, 0 GC cycles, 0s GC pause,
               0.77 MiB heap in use — independent of record count.
```

---

## Platform support

| Platform | `mmap` | 2 MiB aligned | Huge pages | `mlock` | Core affinity |
|---|---|---|---|---|---|
| linux/amd64 | raw `SYS_MMAP` | yes | `MAP_HUGETLB` + `MADV_HUGEPAGE` | yes | `sched_setaffinity` |
| linux/arm64 | raw `SYS_MMAP` | yes | `MAP_HUGETLB` + `MADV_HUGEPAGE` | yes | `sched_setaffinity` |
| darwin/amd64 | raw `SYS_MMAP` | yes (trim) | superpage attempt, usually denied | yes | not exposed by XNU |
| darwin/arm64 | raw `SYS_MMAP` | yes (trim) | no (16 KiB base pages) | yes | not exposed by XNU |
| others | — | — | — | — | `ErrUnsupportedPlatform` |

On darwin, `sysPinThreadToCore` reports `ErrUnsupportedPlatform` rather than pretending: XNU exposes no thread-to-CPU binding, only an affinity *tag* hinting that two threads should share an L2. Reporting success there would be a lie.

Unsupported platforms fail cleanly at construction. **There is deliberately no Go-heap fallback arena** — it would be neither huge-page aligned, nor unswappable, nor invisible to the collector, so it would violate every guarantee the API makes while appearing to work.

---

## Operational requirements

**Reserve huge pages (Linux)** — required for `RequireHugePages`:

```bash
echo 1024 > /proc/sys/vm/nr_hugepages          # 1024 × 2 MiB = 2 GiB
cat /proc/meminfo | grep -i huge
```

**Raise the memory-lock limit** — required for `RequirePinned`:

```bash
ulimit -l unlimited        # or set memlock in /etc/security/limits.conf
# containers: docker run --ulimit memlock=-1:-1 ...
```

**Confirm transparent huge pages are collapsing:**

```bash
grep AnonHugePages /proc/meminfo
```

---

## Design trade-offs and limitations

These are consequences of the design, stated plainly.

**The global ticket counter is the one intentional serialization point.** Every `Publish` performs an atomic increment on a single cache line. A globally ordered ticket sequence cannot be produced without one, and it bounds aggregate ingest regardless of core count. Everything *else* in the data plane is shard-local; this is the exception, and it is deliberate.

**Payload lifetime is caller-managed.** `Record.Payload` aliases a live cell. The instant a consumer callback returns, the slot is released and a producer on another core may overwrite those exact bytes. Use `CopyPayload()` to retain them. This is the price of zero-copy delivery, and it is the sharpest edge in the API.

**Key 0 is reserved** as the index's empty sentinel. `Publish(0, …)` returns `ErrConfig`.

**The index is a hint; the cell header is the authority.** Every lookup validates its answer against the cell it points at, so a recycled slot yields `ErrNotFound` rather than another key's data. Two consequences: a key whose record has aged out of the ring becomes unfindable (correct — the data is genuinely gone), and a key republished at the exact moment its bucket is being reclaimed may be briefly unfindable until its next mutation. The structure is lock-free precisely because it accepts these bounded, detectable staleness windows instead of paying for version counters or epoch reclamation.

**Capacity is fixed at construction.** No growth, no rehashing, no reallocation. That is the definition of memory-bounded, and it means a sustained overload results in backpressure rather than unbounded memory growth.

**One consumer thread per shard.** `Shards` defaults to `GOMAXPROCS`, so a busy engine dedicates that many threads to consumption. On a machine also running producers this oversubscribes cores; size `Shards` to the cores you are willing to give consumers.

**Records in flight at `Close()` are dropped** when no `Consumer` is configured. With a consumer, `Close` drains fully before unmapping — unmapping first would turn an in-flight handler into a `SIGSEGV`, not a Go panic.

**`go vet` requires one exclusion.** Converting an `mmap` result from `uintptr` to `unsafe.Pointer` has no sanctioned pattern in the `unsafeptr` analyzer, which is purely syntactic; the Go runtime performs the identical conversion and is exempt only because it *is* package `runtime`. The conversion is sound — the address is kernel-owned, never Go-heap, and never relocated by the collector:

```bash
go vet -unsafeptr=false ./...     # clean
```

---

## Verification

```bash
go test ./...                     # 27 tests
go test -race ./...               # lock-free protocol under TSan
go test -bench . -cpu=1,2,4,8 ./...
GOOS=linux GOARCH=amd64 go build ./...
```

What the suite actually proves:

- **Layout** — every header offset, header size, cursor separation into distinct cache lines, and shard stride isolation, verified at run time in addition to the compile-time assertions.
- **Arena** — 2 MiB alignment of the returned base, whole-span readability and writability across every page including the final byte, and idempotent teardown.
- **Ring** — strict FIFO order; 500 laps of an 8-cell ring, which is what would expose an ABA or stale sequence; and that the commit index refuses to advance over an unpublished slot while a consumer is denied the record beyond it.
- **Concurrency** — 8 producers × 20,000 records against 2 consumers, asserting no record is lost, none is duplicated, and no payload is torn (each payload encodes its own key and is checked on the way out).
- **Backpressure** — the gate latches above the high-water mark, refuses every write while shut, *stays* shut after a single record is drained (the hysteresis), and releases below the low-water mark. Under concurrent load, consumed count must exactly equal published count.
- **Index reclamation** — 20,000 distinct keys through 512 buckets: 19,447 buckets reclaimed, 41 overflows, and all 33 most-recent keys still resolvable. Without reclamation this case produced ~19,500 overflows and zero resolvable keys.
- **Zero allocation** — `testing.AllocsPerRun` asserts `Publish` + `Lookup` + `Drain` allocates **exactly zero** objects. This test is load-bearing: if the write path allocates, the engine has failed at its stated purpose.

---

## Source map

| File | Lines | Contents |
|---|---:|---|
| `arena.go` | 284 | huge-page alignment, over-allocate-and-trim, prefault, pin |
| `arena_linux.go` | 142 | raw `SYS_MMAP`/`MUNMAP`/`MADVISE`, `MAP_HUGETLB`, `sched_setaffinity` |
| `arena_darwin.go` | 111 | raw `SYS_MMAP`, superpage attempt, honest affinity refusal |
| `arena_unsupported.go` | 50 | clean construction-time failure, no heap fallback |
| `layout.go` | 248 | cell layout, compile-time offset assertions, pointer arithmetic |
| `cacheline.go` | 77 | 128-byte coherence stride, padded cursor type |
| `atomics.go` | 89 | the single choke point for every atomic, alignment contract |
| `ring.go` | 332 | claim / publish / commit-index / dequeue / release turnstiles |
| `index.go` | 409 | lock-free open-addressed key index with dead-bucket reclamation |
| `gate.go` | 119 | hysteretic backpressure valve |
| `fabric.go` | 735 | engine, off-heap control plane, workers, saturation latch |
| `record.go` | 87 | zero-copy record view, FNV-1a checksum |
| `stats.go` | 112 | observability snapshot |
| `cmd/vaultagectl` | 188 | load generator and physical-property reporter |
| `vaultage_test.go` | 847 | 27 tests |
| `bench_test.go` | 246 | benchmarks including the heap baseline |

**Zero dependencies.** `syscall`, `unsafe`, `sync/atomic`, `sync`, `runtime`, `time` — standard library only.
