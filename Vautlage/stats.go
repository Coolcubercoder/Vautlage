package vaultage

// ShardStats is a per-shard snapshot.
type ShardStats struct {
	ID           int
	Capacity     uint64 // cells
	Pending      uint64 // claimed but not yet consumed
	Committed    uint64 // published and awaiting a consumer
	OccupancyPct uint64
	Ingested     uint64
	Consumed     uint64
	Rejected     uint64
	Tail         uint64 // producer cursor
	CommitIndex  uint64 // contiguous committed prefix
	Head         uint64 // consumer cursor
	Arena        ArenaStats
	Index        IndexStats
}

// Stats is an engine-wide snapshot. It is assembled from atomic loads taken at
// slightly different instants, so counters are individually exact but mutually
// approximate. That is a deliberate trade: a globally consistent snapshot would
// require quiescing every core, and observing the engine must never perturb it.
type Stats struct {
	Shards      int
	CellStride  uintptr // bytes per cell
	PayloadCap  uintptr // usable payload bytes per cell
	ArenaBytes  uintptr // total off-heap bytes mapped, including control plane
	PinnedBytes uintptr // off-heap bytes locked into physical RAM
	HugeAligned bool    // every arena base sits on a 2MiB boundary
	HugeNative  bool    // every arena is served from the huge-page pool

	Tickets      uint64 // tickets issued
	Ingested     uint64
	Consumed     uint64
	Rejected     uint64 // rejected for shard fullness
	Dropped      uint64 // tickets issued but not committed
	Pending      uint64
	OccupancyPct uint64

	Gate       GateStats
	ShardStats []ShardStats
}

// Stats returns a snapshot of the engine. Safe to call concurrently with
// ingestion and consumption.
func (v *Vaultage) Stats() Stats {
	s := Stats{
		Shards:      int(v.shardN),
		CellStride:  v.cellStride,
		PayloadCap:  v.payloadCap,
		HugeAligned: true,
		HugeNative:  true,
		Gate:        v.valve.stats(),
		ShardStats:  make([]ShardStats, 0, v.shardN),
	}
	s.Tickets = v.ctl.ticket.load()
	s.Dropped = v.ctl.dropped.load()

	var pending, capacity uint64

	ctl := v.ctlArena.Stats()
	s.ArenaBytes += ctl.Size
	if ctl.Pinned {
		s.PinnedBytes += ctl.Size
	}

	for i := 0; i < int(v.shardN); i++ {
		sh := v.shard(i)
		as := v.arenas[i].Stats()

		ss := ShardStats{
			ID:           i,
			Capacity:     sh.ring.capacity,
			Pending:      sh.ring.pending(),
			Committed:    sh.ring.committed(),
			OccupancyPct: sh.ring.occupancyPct(),
			Ingested:     sh.ingested.load(),
			Consumed:     sh.consumed.load(),
			Rejected:     sh.rejected.load(),
			Tail:         sh.ring.tail.load(),
			CommitIndex:  sh.ring.commit.load(),
			Head:         sh.ring.head.load(),
			Arena:        as,
			Index:        sh.index.stats(),
		}
		s.ShardStats = append(s.ShardStats, ss)

		s.Ingested += ss.Ingested
		s.Consumed += ss.Consumed
		s.Rejected += ss.Rejected
		pending += ss.Pending
		capacity += ss.Capacity

		s.ArenaBytes += as.Size
		if as.Pinned {
			s.PinnedBytes += as.Size
		}
		if !as.HugePageAligned {
			s.HugeAligned = false
		}
		if !as.HugePagesNative {
			s.HugeNative = false
		}
	}

	s.Pending = pending
	if capacity > 0 {
		s.OccupancyPct = pending * 100 / capacity
	}
	return s
}
