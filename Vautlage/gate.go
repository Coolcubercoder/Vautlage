package vaultage

// Gate states. The gate is a single atomic word so that the entire ingestion
// path can test it with one uncontended load, which on both x86-64 and arm64
// costs an L1 hit on a line that is almost always shared-clean.
const (
	gateOpen   uint64 = 0
	gateClosed uint64 = 1
)

// gate is the atomic backpressure valve.
//
// HYSTERESIS. The gate deliberately uses two thresholds rather than one. A
// single threshold produces flapping: occupancy hovers at the limit, the gate
// opens and closes on alternate records, and every producer pays a contended CAS
// on the same line for every operation, which is exactly the false-sharing storm
// the rest of the engine is built to avoid.
//
// With a high-water mark at 90% and a low-water mark at 70%, the gate closes
// once and stays closed until consumers have genuinely reclaimed a fifth of the
// matrix. The CAS on the state word therefore fires roughly twice per
// saturation episode instead of twice per record.
//
// Counters are kept on their own coherence strides so that the statistics do not
// themselves become a contention point.
type gate struct {
	state   padU64 // gateOpen | gateClosed
	closes  padU64 // number of times the gate has latched shut
	opens   padU64 // number of times it has released
	rejects padU64 // ingestion attempts refused while shut

	// highPct/lowPct are immutable after construction.
	highPct uint64
	lowPct  uint64
}

// initGate configures the valve. high must exceed low, and both must be
// percentages in (0,100]; a high mark at or below the low mark would make the
// hysteresis window empty and reintroduce flapping.
func initGate(g *gate, highPct, lowPct uint64) error {
	if highPct == 0 || highPct > 100 || lowPct >= highPct {
		return ErrConfig
	}
	g.highPct = highPct
	g.lowPct = lowPct
	g.state.store(gateOpen)
	return nil
}

// isClosed reports whether ingestion is currently frozen.
//
// This is the single hottest read in the engine: every Publish begins with it.
//
//go:nosplit
func (g *gate) isClosed() bool { return g.state.load() == gateClosed }

// reject records a refused ingestion attempt.
//
//go:nosplit
func (g *gate) reject() { g.rejects.add(1) }

// shut latches the gate closed. It returns true if this call performed the
// transition.
//
// The CAS is what makes the transition exactly-once across all cores: many
// producers may observe saturation simultaneously, but only one wins the
// gateOpen -> gateClosed swap and increments the counter. The losers pay a
// failed CAS and nothing else.
//
// The threshold comparison deliberately does NOT live here. Deciding *whether*
// the engine is saturated requires shard state, and doing that test on this
// line would drag every producer's cache into the gate on every publish. The
// caller tests a shard-local counter and only touches this line on a genuine
// state change.
//
//go:nosplit
func (g *gate) shut() bool {
	if g.state.cas(gateOpen, gateClosed) {
		g.closes.add(1)
		return true
	}
	return false
}

// release reopens the gate. Returns true if this call performed the transition.
//
//go:nosplit
func (g *gate) release() bool {
	if g.state.cas(gateClosed, gateOpen) {
		g.opens.add(1)
		return true
	}
	return false
}

// forceOpen releases the gate unconditionally. Used only during shutdown, so a
// producer blocked on backpressure cannot wedge a draining engine.
func (g *gate) forceOpen() { g.state.store(gateOpen) }

// GateStats reports valve activity.
type GateStats struct {
	Closed        bool   // gate is currently shut
	CloseCount    uint64 // saturation episodes
	OpenCount     uint64 // recoveries
	RejectedCount uint64 // ingestion attempts refused
	HighWaterPct  uint64
	LowWaterPct   uint64
}

func (g *gate) stats() GateStats {
	return GateStats{
		Closed:        g.isClosed(),
		CloseCount:    g.closes.load(),
		OpenCount:     g.opens.load(),
		RejectedCount: g.rejects.load(),
		HighWaterPct:  g.highPct,
		LowWaterPct:   g.lowPct,
	}
}
