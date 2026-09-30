"""
charlib -- loading and analysis for the ESP memory-system characterization.

All numbers in the report come from here, so the notebook stays narrative and the
arithmetic is in one testable place. Run `python3 charlib.py` for a self-check.

Two facts drive most of the arithmetic and are easy to get wrong:

  * Clock domains. The memory tile and the NoC run at 156.25 MHz (the MIG ui_clk);
    accelerator, CPU and IO tiles run at 78.125 MHz. The ratio is exactly 2. Every
    counter in mem.csv ticks at the memory-tile rate, while every cycle count in
    runs.csv is read from the CPU's mcycle and so ticks at the tile rate. Comparing
    them requires the factor of 2 -- see MEM_PER_ACC_CYCLE.

  * The read-channel ceiling. Peak would be one 64-bit beat per memory-tile cycle.
    Both arms saturate at 0.5. Expressing throughput in beats per memory-tile cycle
    makes that ceiling visible as a horizontal line at 0.5 on every plot.
"""

from __future__ import annotations
import os, csv, math, statistics as st
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

ACC_CLK_HZ = 78_125_000       # accelerator / CPU / IO tiles
MEM_CLK_HZ = 156_250_000      # memory tile + NoC (MIG ui_clk)
MEM_PER_ACC_CYCLE = MEM_CLK_HZ / ACC_CLK_HZ   # exactly 2.0
BEAT_BYTES = 8                # DMA_NOC_WIDTH = 64 bit
SIZE_BEATS = 65_536           # beats read per accelerator per run
READ_BEAT_CEILING = 0.5       # measured ceiling, beats per memory-tile cycle

# Categorical palette: slots 1-4 of the validated reference order, unchanged.
# Colour follows the ARM (the entity), never its rank in a filtered view.
# Three arms. Every one of them was built, run on hardware and carries the full
# counter set. A fourth arm with the posted-write barrier disabled was considered
# and is NOT part of this study: it was never synthesised and is not planned, so
# nothing here speaks to what the barrier costs.
PALETTE = {
    "baseline_ot1_gate0":   "#2a78d6",   # slot 1  blue
    "multiot_ot4_gate1":    "#eb6834",   # slot 2  orange
    "multiot_ot2_gate1":    "#1baf7a",   # slot 3  aqua
}
LABEL = {
    "baseline_ot1_gate0": "baseline (no multiOT)",
    "multiot_ot4_gate1":  "multiOT depth 4, barrier on",
    "multiot_ot2_gate1":  "multiOT depth 2, barrier on",
}
# Fixed presentation order, so a missing arm never repaints the others.
ARM_ORDER = ["baseline_ot1_gate0", "multiot_ot4_gate1", "multiot_ot2_gate1"]

INK = {"primary": "#1a1a19", "secondary": "#55544e", "muted": "#8b8a80",
       "grid": "#e6e5e0", "surface": "#ffffff"}


# ---------------------------------------------------------------- loading

@dataclass
class Arm:
    key: str
    label: str
    color: str
    runs: pd.DataFrame
    mem: pd.DataFrame
    tile: pd.DataFrame
    path: str
    dropped: dict = None   # malformed rows discarded per table, by name

    def __repr__(self):
        return (f"<Arm {self.key}: {len(self.runs)} acc rows, "
                f"{len(self.mem)} mem rows, {len(self.tile)} tile rows>")


def _dir_for(key: str) -> str:
    """
    The capture this study uses, one per arm: the `_bp` directory.

    Earlier captures of the same arms exist on disk. They are not used. Two of
    them ran an incomplete sweep (429 accelerator rows instead of 624) and all of
    them predate the backpressure counters, so their mem.csv has 17 columns
    instead of 21. The `_bp` captures are a strict superset: same sweep, same
    bitstreams, more counters. `reproducibility()` compares against them once, as
    evidence that a single capture is not a fluke, and nothing else reads them.
    """
    return os.path.join(RESULTS_DIR, f"run_{key}_bp")


def available_arms() -> list[str]:
    """Arms with a complete, parsed result set, in fixed presentation order."""
    out = []
    for k in ARM_ORDER:
        d = _dir_for(k)
        if all(os.path.isfile(os.path.join(d, f))
               for f in ("runs.csv", "mem.csv", "tile.csv")):
            out.append(k)
    return out


def _read_csv_strict(path: str) -> tuple:
    """
    Read a CSV emitted over the serial link, dropping rows whose column count does
    not match the header.

    A UART capture can lose characters, which merges or truncates a line. Such a
    row is not recoverable and must not be guessed at, so it is dropped and
    counted. The count is surfaced (Arm.dropped) so no analysis silently rests on
    a partial table.
    """
    lines = open(path).read().split("\n")
    hdr = lines[0]
    n = hdr.count(",") + 1
    good = [l for l in lines[1:] if l.strip() and l.count(",") + 1 == n]
    dropped = sum(1 for l in lines[1:] if l.strip() and l.count(",") + 1 != n)
    from io import StringIO
    return pd.read_csv(StringIO("\n".join([hdr] + good))), dropped


def load_arm(key: str) -> Arm:
    d = _dir_for(key)
    runs, d_runs = _read_csv_strict(os.path.join(d, "runs.csv"))
    mem, d_mem = _read_csv_strict(os.path.join(d, "mem.csv"))
    tile, d_tile = _read_csv_strict(os.path.join(d, "tile.csv"))
    a = Arm(key, LABEL.get(key, key), PALETTE.get(key, "#8b8a80"),
            runs, mem, tile, d)
    a.dropped = {"runs": d_runs, "mem": d_mem, "tile": d_tile}
    return a


def load_all() -> dict[str, Arm]:
    return {k: load_arm(k) for k in available_arms()}


# ---------------------------------------------------------------- integrity

def integrity(arm: Arm) -> dict:
    """Checks that must pass before any number from this arm is trusted."""
    r = arm.runs
    n_runs = r["run_id"].nunique()
    per_run = r.groupby(["burst", "rd_per_group", "wr_per_group", "n_active"])
    # repeat spread on the per-run completion time
    spreads = []
    for _, g in per_run:
        v = g.groupby("run_id")["cyc_all_done"].first()
        if len(v) > 1 and v.mean() > 0:
            spreads.append(100.0 * (v.max() - v.min()) / v.mean())
    return {
        "acc_rows": len(r),
        "runs": n_runs,
        "data_errors": int(r["errors"].sum()),
        "timeouts": int(r["timeout"].sum()),
        "configs": int(r.groupby(["burst", "rd_per_group", "wr_per_group"]).ngroups),
        "n_active_levels": sorted(r["n_active"].unique().tolist()),
        "worst_repeat_spread_pct": max(spreads) if spreads else float("nan"),
        "has_start_off": "start_off" in r.columns,
    }


def write_ratio_check(arm: Arm) -> pd.DataFrame:
    """
    Independent proof that the four config registers are not crossed.

    The accelerator writes wr_per_group bursts for every rd_per_group bursts read,
    so ddr_write_beats / n_active should land on SIZE_BEATS * wrg/rdg. If the XML
    <param> order and the conf_info_t declaration order disagreed, socketgen would
    map the registers to the wrong fields and these ratios would be scrambled --
    silently, with no data errors.
    """
    meta = (arm.runs.groupby("run_id")[["n_active", "burst", "rd_per_group", "wr_per_group"]]
            .first())
    m = arm.mem.set_index("run_id").join(meta, how="inner")
    m = m[m["idle"] == 0]
    m["expected_writes"] = m["n_active"] * SIZE_BEATS * m["wr_per_group"] / m["rd_per_group"]
    m["expected_reads"] = m["n_active"] * SIZE_BEATS
    g = (m.groupby(["rd_per_group", "wr_per_group"])
           .apply(lambda d: pd.Series({
               "runs": len(d),
               "write_beats/expected": (d["ddr_write_beats"] / d["expected_writes"].replace(0, float("nan"))).mean(),
               "read_beats/expected": (d["ddr_read_beats"] / d["expected_reads"]).mean(),
               "mean_write_beats": d["ddr_write_beats"].mean(),
           }), include_groups=False))
    return g.reset_index()


def noise_floor(arm: Arm) -> pd.Series:
    """Counters during the idle run -- the background the CPU and Ethernet make."""
    idle = arm.mem[arm.mem["idle"] == 1]
    return idle.iloc[0] if len(idle) else pd.Series(dtype=float)


# ---------------------------------------------------------------- metrics

def _sel(arm: Arm, rdg=1, wrg=1, n_active=None, burst=None):
    r = arm.runs
    m = (r["rd_per_group"] == rdg) & (r["wr_per_group"] == wrg)
    if n_active is not None:
        m &= r["n_active"] == n_active
    if burst is not None:
        m &= r["burst"] == burst
    return r[m]


def completion(arm: Arm, rdg=1, wrg=1) -> pd.DataFrame:
    """
    Mean time for ALL active accelerators to finish, per (burst, n_active).

    cyc_all_done is one value per run, so it is averaged over repeats only -- never
    pooled across accelerators. Pooling cyc_to_done across accelerators mixes the
    systematic between-accelerator spread into what looks like measurement noise.
    """
    d = _sel(arm, rdg, wrg)
    per_run = d.groupby(["burst", "n_active", "run_id"])["cyc_all_done"].first().reset_index()
    out = (per_run.groupby(["burst", "n_active"])["cyc_all_done"]
           .agg(["mean", "min", "max", "count"]).reset_index())
    out["spread_pct"] = 100.0 * (out["max"] - out["min"]) / out["mean"]
    return out


def speedup(base: Arm, other: Arm, rdg=1, wrg=1) -> pd.DataFrame:
    a = completion(base, rdg, wrg).rename(columns={"mean": "base"})[["burst", "n_active", "base"]]
    b = completion(other, rdg, wrg).rename(columns={"mean": "other"})[["burst", "n_active", "other"]]
    m = a.merge(b, on=["burst", "n_active"], how="inner")
    m["speedup"] = m["base"] / m["other"]
    return m


def scaling(arm: Arm, rdg=1, wrg=1) -> pd.DataFrame:
    """n6/n1 -- how much longer six accelerators take than one. 1.0 = perfect."""
    c = completion(arm, rdg, wrg)
    piv = c.pivot(index="burst", columns="n_active", values="mean")
    out = pd.DataFrame(index=piv.index)
    if 1 in piv.columns:
        for n in [x for x in piv.columns if x != 1]:
            out[f"n{n}/n1"] = piv[n] / piv[1]
            out[f"aggregate_gain_n{n}"] = n / (piv[n] / piv[1])
    return out.reset_index()


def read_path(arm: Arm, rdg=1, wrg=1) -> pd.DataFrame:
    """
    Read-channel occupancy and throughput at the memory tile.

    Elapsed is converted from accelerator cycles to memory-tile cycles, because
    ddr_read_* tick at 156.25 MHz while cyc_all_done is counted at 78.125 MHz.
    """
    meta = (arm.runs.groupby("run_id")
            .agg(n_active=("n_active", "first"), burst=("burst", "first"),
                 rdg=("rd_per_group", "first"), wrg=("wr_per_group", "first"),
                 cyc=("cyc_all_done", "first")))
    m = arm.mem.set_index("run_id").join(meta, how="inner")
    m = m[(m["idle"] == 0) & (m["rdg"] == rdg) & (m["wrg"] == wrg)]
    m["elapsed_mem_cyc"] = m["cyc"] * MEM_PER_ACC_CYCLE
    m["busy_frac"] = m["ddr_read_busy"] / m["elapsed_mem_cyc"]
    m["multi_frac"] = m["ddr_read_multi"] / m["elapsed_mem_cyc"]
    m["beats_per_cyc"] = m["ddr_read_beats"] / m["elapsed_mem_cyc"]
    m["read_MBps"] = m["ddr_read_beats"] * BEAT_BYTES / (m["cyc"] / ACC_CLK_HZ) / 1e6
    return (m.groupby(["burst", "n_active"])[
                ["busy_frac", "multi_frac", "beats_per_cyc", "read_MBps",
                 "ddr_read_multi", "ddr_read_busy", "ddr_read_beats"]]
            .mean().reset_index())


def fairness(arm: Arm, burst: int, n_active: int = 6, rep: int = 1,
             rdg=1, wrg=1) -> pd.DataFrame:
    """
    Per-accelerator elapsed time with launch order removed.

    cyc_active = t_done - t_start for that accelerator, so it excludes the CPU's
    sequential launch loop. Present only in runs recorded after the harness gained
    start_off/cyc_active; falls back to cyc_to_done otherwise, which conflates the
    two and should be read with that caveat.
    """
    d = _sel(arm, rdg, wrg, n_active=n_active, burst=burst)
    d = d[d["rep"] == rep].sort_values("acc")
    col = "cyc_active" if "cyc_active" in d.columns else "cyc_to_done"
    out = d[["acc", "acc_y", "acc_x", "hops", col]].copy()
    out = out.rename(columns={col: "elapsed"})
    out.attrs["column_used"] = col
    return out


def fairness_summary(arm: Arm, rdg=1, wrg=1, n_active=6) -> pd.DataFrame:
    rows = []
    for b in sorted(arm.runs["burst"].unique()):
        f = fairness(arm, b, n_active, rdg=rdg, wrg=wrg)
        if len(f) < 2:
            continue
        rows.append({"burst": b, "slowest/fastest": f["elapsed"].max() / f["elapsed"].min(),
                     "fastest": f["elapsed"].min(), "slowest": f["elapsed"].max()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- NoC

# Plane assignment, read off the injection counters rather than from the RTL:
# the memory tile injects one flit per read-data beat on plane 3, and each
# accelerator injects one flit per write-data beat on plane 5.
DMA_RSP_PLANE = 3     # memory tile -> accelerator: read data
DMA_REQ_PLANE = 5     # accelerator -> memory tile: descriptors and write data

TILE_NAME = {0: "mem (0,0)", 1: "cpu (0,1)", 2: "acc0 (0,2)", 3: "acc1 (1,0)",
             4: "acc2 (1,1)", 5: "acc3 (1,2)", 6: "acc4 (2,0)", 7: "acc5 (2,1)",
             8: "io (2,2)"}


def noc(arm: "Arm", rdg=1, wrg=1, n_active=6) -> pd.DataFrame:
    """
    Per-tile NoC link utilization on the two DMA planes.

    `router_p<plane>_d<dir>` counts cycles in which a flit was TRANSMITTED on that
    port -- the RTL drives it from `not data_void_out` (noc32_xy.vhd:282). Despite
    the counter's name, and despite ESP's own print string calling it
    "backpressure cycles", it measures utilization. The signal that would measure
    backpressure, `stop_out or stop_in`, is commented out in the same file, so this
    data cannot say whether anything is congested -- only how much traffic moved.

    Two derived figures, kept separate because they answer different questions:

      util_max_pct    the busiest SINGLE port. Each port is one counter and cannot
                      exceed 100%, so this is the one to read for "is anything
                      near saturation".
      flits_per_cycle the sum over all 5 ports and both planes, divided by 100.
                      A router transmits on several ports in the same cycle, so
                      this legitimately exceeds 1.0; it is throughput, not a
                      percentage of anything.

    Normalised by `dvfs3`, that tile's own free-running cycle counter (every tile
    hardwires mon_dvfs.vf = "1000", so index 3 increments unconditionally), which
    is why no clock conversion is needed: each tile is measured in its own cycles.
    """
    meta = (arm.runs.groupby("run_id")
            .agg(n_active=("n_active", "first"), burst=("burst", "first"),
                 rdg=("rd_per_group", "first"), wrg=("wr_per_group", "first"),
                 cyc=("cyc_all_done", "first")))
    t = arm.tile.set_index("run_id").join(meta, how="inner")
    t = t[(t["idle"] == 0) & (t["rdg"] == rdg) & (t["wrg"] == wrg)
          & (t["n_active"] == n_active)]
    cols = [f"router_p{p}_d{d}" for p in (DMA_REQ_PLANE, DMA_RSP_PLANE) for d in range(5)]
    u = 100.0 * t[cols].div(t["dvfs3"], axis=0)
    # A counter cannot tick more than once per cycle. Anything above 105% is a
    # dropped character inside a number -- structurally valid, numerically corrupt.
    # run_id repeats across tiles, so the index is not unique -- build positionally.
    impossible = (u > 105).any(axis=1).to_numpy()
    u = u.reset_index(drop=True)[~impossible]
    tt = t.reset_index(drop=True)[~impossible]
    out = pd.DataFrame({
        "burst": tt["burst"].to_numpy(),
        "tile": tt["tile"].to_numpy(),
        "util_max_pct": u.max(axis=1).to_numpy(),
        "flits_per_cycle": (u.sum(axis=1) / 100.0).to_numpy(),
    })
    for p in (DMA_REQ_PLANE, DMA_RSP_PLANE):
        pc = [f"router_p{p}_d{d}" for d in range(5)]
        out[f"util_max_p{p}_pct"] = u[pc].max(axis=1).to_numpy()
    g = out.groupby(["burst", "tile"]).mean().reset_index()
    g.attrs["impossible_rows"] = int(impossible.sum())
    return g


def noc_ports(arm: "Arm", tile=0, rdg=1, wrg=1, n_active=6) -> pd.DataFrame:
    """Every individual port counter for one tile -- each is capped at 100%."""
    meta = (arm.runs.groupby("run_id")
            .agg(n_active=("n_active", "first"), burst=("burst", "first"),
                 rdg=("rd_per_group", "first"), wrg=("wr_per_group", "first")))
    t = arm.tile.set_index("run_id").join(meta, how="inner")
    t = t[(t["idle"] == 0) & (t["rdg"] == rdg) & (t["wrg"] == wrg)
          & (t["n_active"] == n_active) & (t["tile"] == tile)]
    rows = []
    for p, pname in ((DMA_RSP_PLANE, "plane 3 - read data out"),
                     (DMA_REQ_PLANE, "plane 5 - requests + write data in")):
        for d in range(5):
            v = 100.0 * (t[f"router_p{p}_d{d}"] / t["dvfs3"])
            v = v[v <= 105]
            if len(v):
                rows.append({"plane": pname,
                             "port": f"d{d}" + (" (local)" if d == 4 else ""),
                             "utilization_pct": v.mean()})
    return pd.DataFrame(rows)


def noc_at_memory(arm: "Arm", rdg=1, wrg=1, n_active=6) -> pd.DataFrame:
    """The memory tile's row, with delivered read throughput alongside."""
    d = noc(arm, rdg, wrg, n_active)
    d = d[d["tile"] == 0].drop(columns="tile")
    rp = read_path(arm, rdg, wrg)
    rp = rp[rp["n_active"] == n_active][["burst", "beats_per_cyc"]]
    return d.merge(rp, on="burst", how="left")


def plot_noc_utilization(arm: "Arm", ax, rdg=1, wrg=1, n_active=6):
    """Busiest single port per tile. Each bar is one counter, so 100% is the cap."""
    import numpy as np
    d = noc(arm, rdg, wrg, n_active).groupby("tile")[
        [f"util_max_p{DMA_RSP_PLANE}_pct", f"util_max_p{DMA_REQ_PLANE}_pct"]].mean()
    x = np.arange(len(d))
    ax.bar(x - 0.2, d[f"util_max_p{DMA_RSP_PLANE}_pct"], 0.38,
           color=PALETTE["baseline_ot1_gate0"], zorder=3,
           label="plane 3 - read data out", linewidth=0.6, edgecolor=INK["surface"])
    ax.bar(x + 0.2, d[f"util_max_p{DMA_REQ_PLANE}_pct"], 0.38,
           color=PALETTE["multiot_ot4_gate1"], zorder=3,
           label="plane 5 - requests + write data in",
           linewidth=0.6, edgecolor=INK["surface"])
    ax.axhline(100, color=INK["muted"], linewidth=1.4, linestyle="--", zorder=2)
    ax.annotate("one flit per cycle - a single port cannot exceed this",
                xy=(0.99, 100), xycoords=("axes fraction", "data"), ha="right",
                va="bottom", fontsize=8, color=INK["muted"])
    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels([TILE_NAME.get(int(i), str(i)) for i in d.index],
                       rotation=45, ha="right", fontsize=8.5)
    _style(ax, ylabel="busiest single port, % of cycles carrying a flit")
    return ax


def plot_noc_ports(arm: "Arm", ax, tile=0, rdg=1, wrg=1, n_active=6):
    """Every port of one tile, so the aggregate is visibly a sum of independent ports."""
    import numpy as np
    d = noc_ports(arm, tile, rdg, wrg, n_active)
    d = d[d["utilization_pct"] > 0.05]
    y = np.arange(len(d))
    colors = [PALETTE["baseline_ot1_gate0"] if "plane 3" in p
              else PALETTE["multiot_ot4_gate1"] for p in d["plane"]]
    ax.barh(y, d["utilization_pct"], 0.62, color=colors, zorder=3,
            linewidth=0.6, edgecolor=INK["surface"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.plane.split(' - ')[0]}  {r.port}" for r in d.itertuples()],
                       fontsize=8.5)
    ax.invert_yaxis()
    for i, v in enumerate(d["utilization_pct"]):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=8.5,
                color=INK["secondary"])
    ax.set_xlim(0, 105)
    _style(ax, xlabel="% of cycles that port carried a flit")
    return ax


# ---------------------------------------------------------------- plotting

def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(INK["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK["grid"])
    ax.grid(True, color=INK["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK["secondary"], labelsize=9)
    if title:
        ax.set_title(title, color=INK["primary"], fontsize=11.5, pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK["secondary"], fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK["secondary"], fontsize=9.5)
    return ax


def plot_burst_sweep(arms: dict, ax, n_active=1, rdg=1, wrg=1, normalize=False):
    """Time to complete vs burst. One line per arm, fixed colour per arm."""
    for k in ARM_ORDER:
        if k not in arms:
            continue
        a = arms[k]
        c = completion(a, rdg, wrg)
        c = c[c["n_active"] == n_active].sort_values("burst")
        if not len(c):
            continue
        y = c["mean"] / c["mean"].iloc[-1] if normalize else c["mean"] / 1000.0
        ax.plot(c["burst"], y, marker="o", markersize=5, linewidth=2,
                color=a.color, label=a.label, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xmargin(0.06); ax.autoscale(enable=True, axis="x")
    _style(ax, ylabel="normalized to largest burst" if normalize
           else "kilocycles to complete (78.125 MHz)",
           xlabel="burst length (64-bit beats per DMA descriptor)")
    return ax


def plot_read_ceiling(arms: dict, ax, n_active=6, rdg=1, wrg=1):
    """Read beats per memory-tile cycle, against the measured 0.5 ceiling."""
    for k in ARM_ORDER:
        if k not in arms:
            continue
        a = arms[k]
        d = read_path(a, rdg, wrg)
        d = d[d["n_active"] == n_active].sort_values("burst")
        if not len(d):
            continue
        ax.plot(d["burst"], d["beats_per_cyc"], marker="o", markersize=5,
                linewidth=2, color=a.color, label=a.label, zorder=3)
    ax.axhline(READ_BEAT_CEILING, color=INK["muted"], linewidth=1.5,
               linestyle="--", zorder=2)
    ax.annotate(f"measured ceiling {READ_BEAT_CEILING} beats/cycle",
                xy=(0.99, READ_BEAT_CEILING), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8.5, color=INK["muted"])
    ax.set_xscale("log", base=2)
    ax.set_xmargin(0.06); ax.autoscale(enable=True, axis="x")
    ax.set_ylim(0, max(0.6, READ_BEAT_CEILING * 1.2))
    _style(ax, ylabel="read beats per memory-tile cycle",
           xlabel="burst length (64-bit beats per DMA descriptor)")
    return ax


def plot_speedup_bars(base: Arm, others: list, ax, n_active=6, rdg=1, wrg=1):
    """Grouped bars: speedup over the baseline, per burst. 1.0 = no change."""
    import numpy as np
    others = [o for o in others if o is not None]
    if not others:
        return ax
    bursts = sorted(speedup(base, others[0], rdg, wrg)["burst"].unique())
    n = len(others)
    width = 0.8 / n
    x = np.arange(len(bursts))
    for i, o in enumerate(others):
        s = speedup(base, o, rdg, wrg)
        s = s[s["n_active"] == n_active].set_index("burst").reindex(bursts)
        off = (i - (n - 1) / 2) * width
        ax.bar(x + off, s["speedup"], width * 0.92, color=o.color,
               label=o.label, zorder=3, linewidth=0.6, edgecolor=INK["surface"])
    ax.axhline(1.0, color=INK["muted"], linewidth=1.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bursts])
    _style(ax, ylabel="speedup over baseline (x)",
           xlabel="burst length (64-bit beats per DMA descriptor)")
    return ax


def plot_occupancy(arm: Arm, ax, n_active=6, rdg=1, wrg=1):
    """
    Stacked read occupancy: share of elapsed time with exactly one read
    outstanding versus two or more. The remainder is time with no read in flight.
    """
    import numpy as np
    d = read_path(arm, rdg, wrg)
    d = d[d["n_active"] == n_active].sort_values("burst")
    x = np.arange(len(d))
    one = (d["busy_frac"] - d["multi_frac"]).clip(lower=0).values
    multi = d["multi_frac"].values
    idle = (1 - d["busy_frac"]).clip(lower=0).values
    ax.bar(x, multi, 0.7, color=PALETTE["multiot_ot4_gate1"], zorder=3,
           label="2 or more reads outstanding", linewidth=0.6, edgecolor=INK["surface"])
    ax.bar(x, one, 0.7, bottom=multi, color=PALETTE["baseline_ot1_gate0"], zorder=3,
           label="exactly 1 read outstanding", linewidth=0.6, edgecolor=INK["surface"])
    ax.bar(x, idle, 0.7, bottom=multi + one, color=INK["grid"], zorder=3,
           label="no read in flight", linewidth=0.6, edgecolor=INK["surface"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(b)) for b in d["burst"]])
    ax.set_ylim(0, 1)
    _style(ax, ylabel="share of elapsed time",
           xlabel="burst length (64-bit beats per DMA descriptor)")
    return ax


def shared_configs(a: "Arm", b: "Arm") -> set:
    """(rdg, wrg, burst) present in BOTH arms -- cross-arm plots must not compare
    a configuration one arm never ran."""
    key = lambda x: set(map(tuple, x.runs[["rd_per_group", "wr_per_group", "burst"]]
                            .drop_duplicates().values))
    return key(a) & key(b)


WORKLOADS = [(1, 1, "1:1 read/write"), (1, 0, "read-only"),
             (4, 1, "4 reads : 1 write"), (1, 4, "1 read : 4 writes")]


def speedup_grid(base: "Arm", others: list, n_active=6) -> "pd.DataFrame":
    """Long-form speedup over baseline, by workload and burst, for the arms given."""
    rows = []
    for o in others:
        shared = shared_configs(base, o)
        for rdg, wrg, name in WORKLOADS:
            s = speedup(base, o, rdg, wrg)
            s = s[s["n_active"] == n_active]
            for _, r in s.iterrows():
                if (rdg, wrg, int(r["burst"])) not in shared:
                    continue
                rows.append({"arm": o.label, "color": o.color, "workload": name,
                             "burst": int(r["burst"]), "speedup": r["speedup"]})
    return pd.DataFrame(rows)


def plot_speedup_grid(base: "Arm", others: list, n_active=6, figsize=(13, 7.5)):
    """
    Speedup over baseline, one panel per workload, bursts on x, arms as colour.

    Only configurations present in BOTH arms are drawn; a panel that is empty for
    an arm says so rather than silently omitting it.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    g = speedup_grid(base, others, n_active)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    for ax, (rdg, wrg, name) in zip(axes.flat, WORKLOADS):
        sub = g[g["workload"] == name]
        bursts = sorted(sub["burst"].unique())
        if not bursts:
            ax.text(0.5, 0.5, f"{name}\nnot measured on both arms",
                    ha="center", va="center", transform=ax.transAxes,
                    color=INK["muted"], fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue
        arms_here = [o for o in others if o.label in set(sub["arm"])]
        w = 0.8 / max(len(arms_here), 1)
        x = np.arange(len(bursts))
        for i, o in enumerate(arms_here):
            d = sub[sub["arm"] == o.label].set_index("burst").reindex(bursts)
            ax.bar(x + (i - (len(arms_here) - 1) / 2) * w, d["speedup"], w * 0.9,
                   color=o.color, label=o.label, zorder=3,
                   linewidth=0.6, edgecolor=INK["surface"])
        ax.axhline(1.0, color=INK["muted"], linewidth=1.4, zorder=2)
        # Same x-extent in every panel, so a workload measured at only two burst
        # sizes reads as sparse coverage rather than as very wide bars.
        ax.set_xlim(-0.7, max(len(bursts), 7) - 0.3)
        ax.set_xticks(x); ax.set_xticklabels([str(b) for b in bursts], fontsize=8.5)
        _style(ax, title=name, xlabel="burst (beats)",
               ylabel="speedup over baseline (x)")
    return fig, axes


def backpressure(arm: "Arm", rdg=1, wrg=1, n_active=6) -> pd.DataFrame:
    """
    Where the memory tile's time goes, as a share of elapsed memory-tile cycles.

    blocked   dma_snd_full  -- read data ready, the NoC will not take it
    starved   dma_rcv_empty -- no request queued, nothing to do
    stop_rsp  the router asserts stop on the DMA response plane
    stop_req  the router asserts stop on the DMA request plane

    These are not mutually exclusive: a tile can be starved of new requests while
    reads it already issued are still in flight, so the columns do not sum to 100.
    They answer a narrower question -- when the tile is not making progress, is it
    because the network refuses the data or because the memory controller has not
    produced it.
    """
    meta = (arm.runs.groupby("run_id")
            .agg(n_active=("n_active", "first"), burst=("burst", "first"),
                 rdg=("rd_per_group", "first"), wrg=("wr_per_group", "first"),
                 cyc=("cyc_all_done", "first")))
    m = arm.mem.set_index("run_id").join(meta, how="inner")
    m = m[(m["idle"] == 0) & (m["rdg"] == rdg) & (m["wrg"] == wrg)
          & (m["n_active"] == n_active)]
    el = m["cyc"] * MEM_PER_ACC_CYCLE
    out = pd.DataFrame({
        "burst": m["burst"],
        "blocked_pct": 100 * m["dma_snd_blocked"] / el,
        "starved_pct": 100 * m["dma_rcv_starved"] / el,
        "stop_rsp_pct": 100 * m["noc_stop_rsp"] / el,
        "stop_req_pct": 100 * m["noc_stop_req"] / el,
        "read_busy_pct": 100 * m["ddr_read_busy"] / el,
    })
    return out.groupby("burst").mean().reset_index()


def plot_backpressure(arm: "Arm", ax, rdg=1, wrg=1, n_active=6):
    """Blocked vs starved vs read-busy against burst size."""
    d = backpressure(arm, rdg, wrg, n_active).sort_values("burst")
    for col, colour, lbl in [
            ("read_busy_pct", PALETTE["multiot_ot2_gate1"], "read outstanding"),
            ("blocked_pct", PALETTE["multiot_ot4_gate1"], "blocked: NoC will not take the data"),
            ("starved_pct", PALETTE["baseline_ot1_gate0"], "starved: no request to work on"),
            ("stop_rsp_pct", INK["muted"], "router stop on response plane")]:
        ax.plot(d["burst"], d[col], marker="o", markersize=5, linewidth=2,
                color=colour, label=lbl, zorder=3,
                linestyle="--" if col == "stop_rsp_pct" else "-")
    ax.set_xscale("log", base=2)
    ax.set_xmargin(0.06); ax.autoscale(enable=True, axis="x")
    ax.set_ylim(0, 100)
    _style(ax, ylabel="share of elapsed memory-tile cycles (%)",
           xlabel="burst length (64-bit beats per DMA descriptor)")
    return ax


def legend(ax, ncol=1):
    """
    Legend for 2+ series only. A single-series chart is identified by its title,
    so a one-entry legend box is noise. Stacked-segment charts always get one,
    because the segments are not otherwise distinguishable.
    """
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) < 2:
        return None
    return ax.legend(frameon=False, fontsize=9, labelcolor=INK["secondary"], ncol=ncol)


# ---------------------------------------------------------------- self-check

if __name__ == "__main__":
    arms = load_all()
    print(f"results dir : {RESULTS_DIR}")
    print(f"arms found  : {list(arms)}\n")
    for k, a in arms.items():
        print(f"--- {a.label}  ({a.key})")
        it = integrity(a)
        for kk, vv in it.items():
            print(f"      {kk:26s} {vv}")
        print("    register cross-check (write beats / expected):")
        print(write_ratio_check(a).to_string(index=False).replace("\n", "\n      "))
        print()
    if "baseline_ot1_gate0" in arms:
        base = arms["baseline_ot1_gate0"]
        print("--- baseline scaling (n6/n1, 1:1)")
        print(scaling(base).to_string(index=False))
        for other in [k for k in ARM_ORDER if k != "baseline_ot1_gate0" and k in arms]:
            print(f"\n--- speedup: {arms[other].label}, n_active=6, 1:1")
            s = speedup(base, arms[other])
            print(s[s["n_active"] == 6].to_string(index=False))


# ---------------------------------------------------------------- the rate quantum

def occupancy(arm: Arm) -> pd.DataFrame:
    """
    One row per run: throughput and every cycle-occupancy counter, in consistent
    units. This is the table every conclusion about the memory path rests on.

    Elapsed time comes from the CPU's cycle count, which ticks at the accelerator
    rate, so it is converted to memory-tile cycles before being used as the
    denominator for counters that tick at the memory-tile rate.
    """
    r, m = arm.runs, arm.mem
    agg = r.groupby("run_id").agg(
        n_active=("n_active", "first"), burst=("burst", "first"),
        rd=("rd_per_group", "first"), wr=("wr_per_group", "first"),
        rep=("rep", "first"), cyc_acc=("cyc_all_done", "max"),
    ).reset_index()
    mm = m.groupby("run_id").sum(numeric_only=True).reset_index()
    d = agg.merge(mm, on="run_id", suffixes=("", "_m"))
    d["mem_cyc"] = d.cyc_acc * MEM_PER_ACC_CYCLE
    d["rd_beats_per_cyc"] = d.ddr_read_beats / d.mem_cyc
    d["wr_beats_per_cyc"] = d.ddr_write_beats / d.mem_cyc
    d["tot_beats_per_cyc"] = (d.ddr_read_beats + d.ddr_write_beats) / d.mem_cyc
    # Per accelerator, in ITS own clock. 1.0 means one 64-bit beat per
    # accelerator cycle, which is that port's architectural maximum.
    d["words_per_acc_cyc"] = (d.ddr_read_beats + d.ddr_write_beats) / d.cyc_acc
    for c in ("ddr_read_busy", "ddr_read_multi", "dma_snd_blocked",
              "dma_rcv_starved", "noc_stop_rsp", "noc_stop_req"):
        if c in d:
            d[c + "_pct"] = 100 * d[c] / d.mem_cyc
    # Response-plane stops per read beat actually delivered. At 1.0 the tile is
    # stopped once for every beat it ships -- a receipt, not a constraint.
    d["stops_per_beat"] = (d.noc_stop_rsp / d.ddr_read_beats).replace(
        [np.inf, -np.inf], np.nan)
    d["arm"] = arm.key
    return d


def occupancy_all(arms: dict) -> pd.DataFrame:
    """occupancy() for every arm, concatenated, reps collapsed by mean."""
    d = pd.concat([occupancy(a) for a in arms.values()], ignore_index=True)
    keys = ["arm", "rd", "wr", "n_active", "burst"]
    return d.groupby(keys).mean(numeric_only=True).reset_index()


def per_accelerator_share(arms: dict, burst: int = 16384, rdg=1, wrg=0) -> pd.DataFrame:
    """
    Aggregate throughput split by the number of accelerators sharing it.

    If the memory path were a pool the accelerators draw from independently, the
    aggregate would grow with n_active. If instead one server is shared, the
    aggregate is flat and the per-accelerator share falls as 1/n. This table
    distinguishes the two by inspection.
    """
    g = occupancy_all(arms)
    s = g[(g.burst == burst) & (g.rd == rdg) & (g.wr == wrg)].copy()
    s["per_acc"] = s.tot_beats_per_cyc / s.n_active
    s["label"] = s.arm.map(LABEL)
    return s[["label", "n_active", "tot_beats_per_cyc", "per_acc",
              "words_per_acc_cyc", "noc_stop_rsp_pct"]].sort_values(
        ["label", "n_active"]).reset_index(drop=True)


def reproducibility() -> pd.DataFrame:
    """
    Compare each arm's capture against an earlier, independent capture of the
    same sweep on the same bitstream.

    This is the only place the superseded directories are read. It answers one
    question -- is a single capture stable enough to draw conclusions from -- and
    is not part of the analysis proper.
    """
    pairs = [("baseline_ot1_gate0", "run_baseline_ot1_gate0_bp",
              "run_baseline_ot1_gate0_v2"),
             ("multiot_ot4_gate1", "run_multiot_ot4_gate1_bp",
              "run_multiot_ot4_gate1_v2"),
             ("multiot_ot2_gate1", "run_multiot_ot2_gate1_bp",
              "run_multiot_ot2_gate1")]

    def series(d):
        r, _ = _read_csv_strict(os.path.join(RESULTS_DIR, d, "runs.csv"))
        m, _ = _read_csv_strict(os.path.join(RESULTS_DIR, d, "mem.csv"))
        a = r.groupby("run_id").agg(
            n=("n_active", "first"), b=("burst", "first"),
            rd=("rd_per_group", "first"), wr=("wr_per_group", "first"),
            cyc=("cyc_all_done", "max")).reset_index()
        mm = m.groupby("run_id").sum(numeric_only=True).reset_index()
        j = a.merge(mm, on="run_id")
        j["tot"] = (j.ddr_read_beats + j.ddr_write_beats) / (j.cyc * MEM_PER_ACC_CYCLE)
        return j.groupby(["n", "b", "rd", "wr"]).tot.mean()

    out = []
    for key, new, old in pairs:
        if not os.path.isdir(os.path.join(RESULTS_DIR, old)):
            continue
        A, B = series(new), series(old)
        j = pd.concat([A.rename("used"), B.rename("earlier")], axis=1).dropna()
        rel = (j.used - j.earlier).abs() / j.earlier * 100
        out.append({"arm": LABEL[key], "matched points": len(j),
                    "median |diff| %": round(rel.median(), 3),
                    "max |diff| %": round(rel.max(), 2)})
    return pd.DataFrame(out)


def plot_bp_vs_throughput(arms: dict, n_active: int = 6, figsize=(11.5, 7.2)):
    """
    Throughput and response-plane stops per delivered beat, on a shared burst
    axis, for read-only and balanced traffic.

    The two rows together are the argument that backpressure is a consequence of
    delivery rather than a cause of the limit: throughput peaks where stops per
    beat is near zero and collapses to the quantum exactly as it reaches one.
    """
    import matplotlib.pyplot as plt
    g = occupancy_all(arms)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    for col, (rdg, wrg, title) in enumerate(
            [(1, 0, "read-only  (rd:wr = 1:0)"), (1, 1, "balanced  (rd:wr = 1:1)")]):
        ax0, ax1 = axes[0][col], axes[1][col]
        for arm in ARM_ORDER:
            s = g[(g.arm == arm) & (g.rd == rdg) & (g.wr == wrg) &
                  (g.n_active == n_active)].sort_values("burst")
            if s.empty:
                continue
            ax0.plot(s.burst, s.tot_beats_per_cyc, "o-", color=PALETTE[arm],
                     label=LABEL[arm], lw=2, ms=5)
            ax1.plot(s.burst, s.stops_per_beat, "o-", color=PALETTE[arm], lw=2, ms=5)
        ax0.axhline(READ_BEAT_CEILING, color=INK["muted"], ls=":", lw=1.2)
        ax0.text(16384, READ_BEAT_CEILING - 0.03,
                 "0.500  =  one 64-bit beat per accelerator cycle",
                 fontsize=7.5, color=INK["secondary"], va="top", ha="right")
        ax1.axhline(1.0, color=INK["muted"], ls=":", lw=1.2)
        ax1.text(9, 1.04, "one stop per delivered beat", fontsize=7.5,
                 color=INK["secondary"], va="bottom")
        ax0.set_title(title, fontsize=10.5, color=INK["primary"], loc="left")
        for ax in (ax0, ax1):
            ax.set_xscale("log", base=2)
            ax.grid(alpha=.35, color=INK["grid"])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        ax0.set_ylim(0, 0.95)
        ax1.set_ylim(0, 1.18)
        ax1.set_xlabel("burst length (64-bit beats per DMA descriptor)", fontsize=9)
        if col == 0:
            ax0.set_ylabel("throughput\n(beats per memory-tile cycle)", fontsize=9)
            ax1.set_ylabel("response-plane stops\nper delivered read beat", fontsize=9)
            ax0.legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"Backpressure is highest exactly where throughput is lowest   "
                 f"·   {n_active} accelerators",
                 fontsize=12, color=INK["primary"], x=0.02, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return fig
