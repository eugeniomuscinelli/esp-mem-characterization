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


# Two campaigns share this code. The 3x3 is the completed study; the 4x4 uses
# the same three arms on a larger mesh with the corrected counters, and lands in
# a separate tree so neither can overwrite the other.
CAMPAIGNS = {
    "3x3": {
        "root": RESULTS_DIR,
        "dirs": {"baseline_ot1_gate0": "run_baseline_ot1_gate0_bp",
                 "multiot_ot4_gate1":  "run_multiot_ot4_gate1_bp",
                 "multiot_ot2_gate1":  "run_multiot_ot2_gate1_bp"},
    },
    "4x4": {
        # Beside charlib.py when the campaign travels with it (a repo checkout, a
        # Colab VM); the working tree on the machine that ran it otherwise.
        "root": (RESULTS_DIR if os.path.isdir(os.path.join(RESULTS_DIR, "run_4x4_baseline"))
                 else os.path.expanduser("~/char_results_4x4")),
        "dirs": {"baseline_ot1_gate0": "run_4x4_baseline",
                 "multiot_ot4_gate1":  "run_4x4_ot4",
                 "multiot_ot2_gate1":  "run_4x4_ot2"},
    },
}


def _dir_for(key: str, campaign: str = "3x3") -> str:
    """
    The capture a campaign uses for one arm.

    For the 3x3 these are the `_bp` directories. Earlier captures of the same
    arms exist on disk and are not used: two ran an incomplete sweep and all of
    them predate the backpressure counters. `reproducibility()` compares against
    them once, and nothing else reads them.
    """
    c = CAMPAIGNS[campaign]
    return os.path.join(c["root"], c["dirs"][key])


def available_arms(campaign: str = "3x3") -> list[str]:
    """Arms with a complete, parsed result set, in fixed presentation order."""
    out = []
    for k in ARM_ORDER:
        if k not in CAMPAIGNS[campaign]["dirs"]:
            continue
        d = _dir_for(k, campaign)
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


def load_arm(key: str, campaign: str = "3x3") -> Arm:
    d = _dir_for(key, campaign)
    runs, d_runs = _read_csv_strict(os.path.join(d, "runs.csv"))
    mem, d_mem = _read_csv_strict(os.path.join(d, "mem.csv"))
    tile, d_tile = _read_csv_strict(os.path.join(d, "tile.csv"))
    a = Arm(key, LABEL.get(key, key), PALETTE.get(key, "#8b8a80"),
            runs, mem, tile, d)
    a.dropped = {"runs": d_runs, "mem": d_mem, "tile": d_tile}
    return a


def load_all(campaign: str = "3x3") -> dict[str, Arm]:
    return {k: load_arm(k, campaign) for k in available_arms(campaign)}


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

    The denominator matters. Every counter here ticks at the memory-tile rate, so
    an occupancy is only exact if the elapsed count shares that clock AND the same
    sampling window. The 4x4 campaign added `mem_cycles`, a free-running counter
    in the monitor's own domain latched by the same burst write as the others, so
    it satisfies both and is used when present. The 3x3 has no such column and
    falls back to the CPU's cycle count scaled by the clock ratio, which shares
    neither exactly -- which is why its starvation occupancies can exceed 100%.
    `mem_cyc_source` records which was used.
    """
    r, m = arm.runs, arm.mem
    agg = r.groupby("run_id").agg(
        n_active=("n_active", "first"), burst=("burst", "first"),
        rd=("rd_per_group", "first"), wr=("wr_per_group", "first"),
        rep=("rep", "first"), cyc_acc=("cyc_all_done", "max"),
    ).reset_index()
    mm = m.groupby("run_id").sum(numeric_only=True).reset_index()
    d = agg.merge(mm, on="run_id", suffixes=("", "_m"))
    if "mem_cycles" in d.columns and (d.mem_cycles > 0).all():
        d["mem_cyc"] = d.mem_cycles.astype(float)
        d["mem_cyc_source"] = "measured"
    else:
        d["mem_cyc"] = d.cyc_acc * MEM_PER_ACC_CYCLE
        d["mem_cyc_source"] = "derived"
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


# ---------------------------------------------------------------- 4x4 figures

def _ax(ax, xlabel=None, ylabel=None, title=None, logx=False):
    """Recessive axes: thin grid, no top/right spines, ink-token text."""
    if logx:
        ax.set_xscale("log", base=2)
    ax.grid(alpha=.3, color=INK["grid"], lw=.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK["grid"])
    ax.tick_params(colors=INK["secondary"], labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK["secondary"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK["secondary"])
    if title:
        ax.set_title(title, fontsize=10.5, color=INK["primary"], loc="left", pad=10)
    return ax


def _label_end(ax, x, y, text, color):
    """Direct label in ink, placed beside the final marker which carries the hue."""
    ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=8.5, color=INK["secondary"])
    ax.plot([x], [y], "o", color=color, ms=7, zorder=5)


def plot_speedup_vs_fanin(arms, figsize=(12.0, 4.3)):
    """
    Speedup over baseline against the number of concurrent accelerators.

    The question the larger mesh was built to answer, so it gets the simplest
    possible form: one ordered x-axis, one line per arm, a reference line at
    parity. Two panels because the answer differs between read-only and balanced
    traffic, and a single panel would hide that.
    """
    import matplotlib.pyplot as plt
    g = occupancy_all(arms)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    panels = [(1, 0, 8, "read-only, 8-beat descriptors"),
              (1, 1, 32, "balanced 1:1, 32-beat descriptors")]
    for ax, (rdg, wrg, burst, title) in zip(axes, panels):
        base = g[(g.arm == "baseline_ot1_gate0") & (g.rd == rdg) & (g.wr == wrg)
                 & (g.burst == burst)].set_index("n_active").tot_beats_per_cyc
        ax.axhline(1.0, color=INK["muted"], ls=":", lw=1.2, zorder=1)
        for arm in ("multiot_ot2_gate1", "multiot_ot4_gate1"):
            s = g[(g.arm == arm) & (g.rd == rdg) & (g.wr == wrg)
                  & (g.burst == burst)].set_index("n_active").tot_beats_per_cyc
            sp = (s / base).sort_index()
            ax.plot(sp.index, sp.values, "-o", color=PALETTE[arm], lw=2, ms=7,
                    label=LABEL[arm], zorder=3)
            _label_end(ax, sp.index[-1], sp.values[-1],
                       f"{sp.values[-1]:.2f}x", PALETTE[arm])
        _ax(ax, "concurrently active accelerators", None, title)
        ax.set_xticks(sorted(g.n_active.unique()))
        ax.set_xlim(0.2, 15.6)
    axes[0].set_ylabel("throughput relative to baseline", fontsize=9,
                       color=INK["secondary"])
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    for ax in axes:
        ax.axvline(6, color=INK["muted"], ls="--", lw=1, zorder=1)
    axes[0].annotate("the 3x3 campaign\nmeasured only to here", xy=(6.35, 1.18),
                     fontsize=8, color=INK["muted"], ha="left", va="bottom")
    fig.suptitle("multiOT's benefit grows with fan-in, then plateaus",
                 fontsize=12.5, color=INK["primary"], x=0.012, ha="left", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_inbound_onset(arms, burst=256, figsize=(7.6, 4.3)):
    """
    Inbound request-plane stall against accelerator count.

    One panel, two workload categories. The point is a threshold, so the x-axis
    is the ordered variable and the 3x3's range is marked rather than described.
    """
    import matplotlib.pyplot as plt
    g = occupancy_all(arms)
    b = g[(g.arm == "baseline_ot1_gate0") & (g.burst == burst)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvspan(0.2, 6, color=INK["grid"], alpha=.55, zorder=0, lw=0)
    ax.annotate("range covered\nby the 3x3 campaign", xy=(2.3, 88), fontsize=8,
                color=INK["muted"], ha="center", va="center")
    for (rdg, wrg, lab, col) in [(1, 0, "read-only", PALETTE["baseline_ot1_gate0"]),
                                 (1, 1, "balanced 1:1", PALETTE["multiot_ot4_gate1"])]:
        s = b[(b.rd == rdg) & (b.wr == wrg)].sort_values("n_active")
        ax.plot(s.n_active, s.noc_stop_req_pct, "-o", color=col, lw=2, ms=7,
                label=lab, zorder=3)
        _label_end(ax, s.n_active.iloc[-1], s.noc_stop_req_pct.iloc[-1],
                   f"{s.noc_stop_req_pct.iloc[-1]:.0f}%", col)
    _ax(ax, "concurrently active accelerators",
        "cycles the memory tile refused\nan inbound request flit (%)")
    ax.set_xticks(sorted(g.n_active.unique()))
    ax.set_xlim(0.2, 15.6); ax.set_ylim(-3, 105)
    ax.legend(fontsize=8.5, frameon=False, loc="center left")
    ax.set_title(f"Inbound backpressure against fan-in  ·  "
                 f"baseline, {burst}-beat descriptors",
                 fontsize=11, color=INK["primary"], loc="left", pad=10)
    ax.annotate("read-only: exactly 0.0%\nthrough 6 accelerators", xy=(4.3, 4),
                fontsize=8, color=INK["secondary"], ha="center", va="bottom")
    fig.tight_layout()
    return fig


def plot_quantum(arms, burst=16384, figsize=(12.0, 4.3)):
    """
    The central measurement, in both units.

    Left: aggregate throughput, which is flat in the number of accelerators.
    Right: the same divided by the accelerator clock, which lands on 1.000.
    Two panels rather than two y-axes, because a dual-axis chart would imply a
    relationship between the scales that does not exist.
    """
    import matplotlib.pyplot as plt
    g = occupancy_all(arms)
    s = g[(g.rd == 1) & (g.wr == 0) & (g.burst == burst)]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # The three arms are identical to four decimal places at this operating point,
    # so drawn at equal weight only the last would be visible. Nesting the widths
    # shows the coincidence instead of hiding two thirds of the data under one line.
    width = {ARM_ORDER[0]: (6.0, 13), ARM_ORDER[1]: (3.0, 9), ARM_ORDER[2]: (1.4, 5)}
    for z, arm in enumerate(ARM_ORDER):
        t = s[s.arm == arm].sort_values("n_active")
        if t.empty:
            continue
        lw, ms = width[arm]
        for ax, col in ((axes[0], "tot_beats_per_cyc"), (axes[1], "words_per_acc_cyc")):
            ax.plot(t.n_active, t[col], "-o", color=PALETTE[arm], lw=lw, ms=ms,
                    label=LABEL[arm], zorder=3 + z, alpha=.95,
                    markeredgecolor="white", markeredgewidth=.8)
    axes[0].axhline(READ_BEAT_CEILING, color=INK["muted"], ls=":", lw=1.2)
    axes[0].annotate("0.500", xy=(13, READ_BEAT_CEILING), xytext=(0, 5),
                     textcoords="offset points", fontsize=8.5, color=INK["muted"],
                     ha="right")
    axes[1].axhline(1.0, color=INK["muted"], ls=":", lw=1.2)
    _ax(axes[0], "concurrently active accelerators",
        "aggregate throughput\n(beats per memory-tile cycle)",
        "Aggregate is flat: 13 accelerators deliver what 1 delivers")
    _ax(axes[1], "concurrently active accelerators",
        "per accelerator clock cycle\n(64-bit words)",
        "The quantum: one 64-bit word per accelerator cycle")
    for ax in axes:
        ax.set_xticks(sorted(g.n_active.unique()))
        ax.set_xlim(0.2, 14)
    axes[0].set_ylim(0, 0.62); axes[1].set_ylim(0.97, 1.03)
    axes[0].legend(fontsize=8.5, frameon=False, loc="lower right",
                   title="all three coincide", title_fontsize=8.5)
    axes[0].get_legend().get_title().set_color(INK["muted"])
    fig.tight_layout()
    return fig


def plot_fairness_by_index(arms, burst=256, n_active=13, figsize=(8.6, 4.3)):
    """
    Per-accelerator elapsed time at full contention, ordered by accelerator index.

    Identity on the x-axis is categorical but ordered, and the claim is about that
    ordering, so a line over the index reads better than grouped bars.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    for k in ARM_ORDER:
        if k not in arms:
            continue
        r = arms[k].runs
        s = r[(r.n_active == n_active) & (r.burst == burst)
              & (r.rd_per_group == 1) & (r.wr_per_group == 1)]
        per = s.groupby("acc").cyc_active.mean()
        rel = per / per.min()
        ax.plot(rel.index, rel.values, "-o", color=PALETTE[k], lw=2, ms=7,
                label=LABEL[k], zorder=3)
    ax.axhline(1.0, color=INK["muted"], ls=":", lw=1.2)
    _ax(ax, "accelerator index", "elapsed time relative to\nthe fastest accelerator",
        f"Service is ordered by accelerator index  ·  "
        f"{n_active} accelerators, {burst}-beat descriptors, 1:1")
    ax.set_xticks(range(0, n_active))
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_inbound_not_the_limit(arms, n_active=13, figsize=(7.8, 5.4)):
    """
    Throughput and inbound stall on a shared descriptor axis.

    Two rows rather than two y-axes: the claim is that one of these varies while
    the other does not, and a dual-axis chart would invite reading a relationship
    off the crossing point instead.
    """
    import matplotlib.pyplot as plt
    g = occupancy_all(arms)
    s = g[(g.arm == "baseline_ot1_gate0") & (g.rd == 1) & (g.wr == 0)
          & (g.n_active == n_active)].sort_values("burst")
    col = PALETTE["baseline_ot1_gate0"]
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].plot(s.burst, s.tot_beats_per_cyc, "-o", color=col, lw=2, ms=7, zorder=3)
    axes[0].axhline(READ_BEAT_CEILING, color=INK["muted"], ls=":", lw=1.2)
    axes[0].annotate("0.500", xy=(16384, READ_BEAT_CEILING), xytext=(0, 5),
                     textcoords="offset points", fontsize=8.5,
                     color=INK["muted"], ha="right")
    axes[1].plot(s.burst, s.noc_stop_req_pct, "-o", color=col, lw=2, ms=7, zorder=3)
    lo, hi = s[s.burst <= 64].noc_stop_req_pct.min(), s[s.burst <= 64].noc_stop_req_pct.max()
    axes[1].axhspan(lo, hi, color=INK["grid"], alpha=.7, zorder=0, lw=0)
    axes[1].annotate(f"{lo:.1f}% to {hi:.1f}% across bursts 8-64,\n"
                     f"while throughput over the same points varies "
                     f"{s[s.burst<=64].tot_beats_per_cyc.max()/s[s.burst<=64].tot_beats_per_cyc.min():.1f}x",
                     xy=(300, (lo + hi) / 2 - 22), fontsize=8.5,
                     color=INK["secondary"], ha="left", va="top")
    _ax(axes[0], None, "throughput\n(beats per memory-tile cycle)", logx=True)
    _ax(axes[1], "descriptor length (64-bit beats)",
        "inbound request flits\nrefused (% of cycles)", logx=True)
    axes[0].set_ylim(0, 0.68)
    axes[1].set_ylim(0, 108)
    fig.suptitle(f"Inbound stall is pinned while throughput varies  ·  "
                 f"baseline, {n_active} accelerators, read-only",
                 fontsize=11, color=INK["primary"], x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------- run anatomy

SIZE_BEATS_4X4 = 65_536          # read beats per accelerator per run, every run
PLM_WORDS      = 16_384          # the accelerator's local buffer, 64-bit words
FLITS_PER_DESC = 3               # header, address, length -- independent of length
MIG_PORT_BPS   = 8 * 156.25e6    # 64 bit at ui_clk = 1.25 GB/s
DDR4_BPS       = 10.0e9          # 1250 MT/s x 64 bit


def run_anatomy(arm: Arm, n_active: int = 13) -> pd.DataFrame:
    """
    What one run consists of, per accelerator, at each descriptor length.

    Every run moves the same read volume; only the number of requests changes.
    `flits` is measured, not assumed, so the 3-flits-per-descriptor claim is
    checked against hardware rather than stated.
    """
    o = occupancy(arm)
    s = o[(o.rd == 1) & (o.wr == 0) & (o.n_active == n_active)].groupby("burst").mean(
        numeric_only=True)
    d = pd.DataFrame(index=s.index)
    d["descriptors per accelerator"] = (SIZE_BEATS_4X4 // s.index).astype(int)
    d["PLM words used"] = s.index.map(lambda b: min(b, PLM_WORDS)).astype(int)
    d["PLM fraction used"] = (d["PLM words used"] / PLM_WORDS).round(4)
    d["request flits measured"] = s.dma_req_flits.astype(int)
    d["flits per descriptor"] = (s.dma_req_flits
                                 / (d["descriptors per accelerator"] * n_active)).round(2)
    d.index.name = "descriptor length (beats)"
    return d


def traffic_by_mix(arm: Arm, n_active: int = 13, burst: int = 256) -> pd.DataFrame:
    """
    Total DRAM traffic per accelerator per run, by read:write ratio.

    The read volume is fixed by SIZE_BEATS; writes are a multiple of it set by the
    ratio. So the mixes do NOT move equal bytes, which matters whenever they are
    compared by throughput.
    """
    o = occupancy(arm)
    s = o[(o.n_active == n_active) & (o.burst == burst)]
    rows = []
    for (rd, wr), t in s.groupby(["rd", "wr"]):
        t = t.mean(numeric_only=True)
        rb, wb = t.ddr_read_beats / n_active, t.ddr_write_beats / n_active
        rows.append({"mix": f"{int(rd)}:{int(wr)}",
                     "read beats": round(rb), "write beats": round(wb),
                     "total beats": round(rb + wb),
                     "KiB per accelerator": round((rb + wb) * 8 / 1024),
                     "PLM capacities": round((rb + wb) / PLM_WORDS, 2)})
    return pd.DataFrame(rows).set_index("mix")


def bottleneck_ladder(arms: dict, n_active: int = 13) -> pd.DataFrame:
    """
    Every candidate limit in the path, with the utilisation actually reached.

    A resource cannot be the constraint while it is idle, so this table is the
    argument: only the row whose utilisation approaches its own maximum is a
    candidate, and the mix dependence distinguishes a direction-blind resource
    (the AXI port) from a direction-specific one (the NoC path).
    """
    g = occupancy_all(arms)
    rows = []
    long_ro = g[(g.rd == 1) & (g.wr == 0) & (g.burst == 16384)
                & (g.n_active == n_active)].tot_beats_per_cyc.mean()
    peak = g.tot_beats_per_cyc.max()
    ro_peak = g[(g.rd == 1) & (g.wr == 0)].tot_beats_per_cyc.max()
    rows.append({"resource": "DDR4 chip", "capacity": f"{DDR4_BPS/1e9:.0f} GB/s",
                 "reached at long descriptors": f"{100*long_ro*MIG_PORT_BPS/DDR4_BPS:.1f}%",
                 "best reached anywhere": f"{100*peak*MIG_PORT_BPS/DDR4_BPS:.1f}%"})
    rows.append({"resource": "MIG AXI port", "capacity": f"{MIG_PORT_BPS/1e9:.2f} GB/s",
                 "reached at long descriptors": f"{100*long_ro:.1f}%",
                 "best reached anywhere": f"{100*peak:.1f}%"})
    rows.append({"resource": "outbound NoC path (read data)",
                 "capacity": "not independently known",
                 "reached at long descriptors": f"{100*long_ro:.1f}%",
                 "best reached anywhere": f"{100*ro_peak:.1f}% (read-only ceiling)"})
    return pd.DataFrame(rows).set_index("resource")


def elapsed_comparison(arms: dict, rdg=1, wrg=0, n_active: int = 13) -> pd.DataFrame:
    """
    Total elapsed cycles at constant data moved: who finishes first.

    The most direct statement the campaign can make, and the one that shows the
    optimum is neither the shortest nor the longest descriptor.
    """
    cols = {}
    for k in ARM_ORDER:
        if k not in arms:
            continue
        r = arms[k].runs
        s = r[(r.n_active == n_active) & (r.rd_per_group == rdg)
              & (r.wr_per_group == wrg)]
        cols[LABEL[k]] = (s.groupby(["burst", "rep"]).cyc_all_done.max()
                          .groupby("burst").mean())
    d = pd.DataFrame(cols)
    base = d[LABEL["baseline_ot1_gate0"]]
    d.insert(0, "baseline, ms", (base / ACC_CLK_HZ * 1e3).round(2))
    d["vs baseline optimum"] = (base / base.min()).round(2)
    d.index.name = "descriptor length (beats)"
    return d.round(0)


def service_time(arms: dict, rdg=1, wrg=0, n_active: int = 13) -> pd.DataFrame:
    """
    Cycles the memory tile spends on one descriptor, and the inbound refusal it
    produces while it does.

    Inbound stall is service time seen from the network side: the tile accepts a
    descriptor's flits, then refuses everything offered until it is done.
    """
    out = {}
    for k in ARM_ORDER:
        if k not in arms:
            continue
        o = occupancy(arms[k])
        s = o[(o.rd == rdg) & (o.wr == wrg) & (o.n_active == n_active)].groupby(
            "burst").mean(numeric_only=True)
        rate = s.ddr_read_beats / s.mem_cyc
        out[(LABEL[k], "service cycles")] = (s.index / rate).round(0)
        out[(LABEL[k], "inbound stall %")] = (100 * s.noc_stop_req / s.mem_cyc).round(1)
    d = pd.DataFrame(out)
    d.columns = pd.MultiIndex.from_tuples(d.columns)
    d.index.name = "descriptor length (beats)"
    return d


def effective_concurrency(arms: dict, rdg=1, wrg=0, n_active: int = 13) -> pd.DataFrame:
    """
    Aggregate throughput expressed in units of one accelerator's port rate.

    One accelerator absorbs at most one 64-bit word per its own 78.125 MHz cycle,
    so this ratio reads as how many accelerators were effectively being served at
    once. It is bounded above by n_active everywhere in the campaign, which is the
    check that makes the reading safe rather than assumed.
    """
    out = {}
    for k in ARM_ORDER:
        if k not in arms:
            continue
        o = occupancy(arms[k])
        s = o[(o.rd == rdg) & (o.wr == wrg) & (o.n_active == n_active)]
        out[LABEL[k]] = s.groupby("burst").words_per_acc_cyc.mean().round(3)
    d = pd.DataFrame(out)
    d.index.name = "descriptor length (beats)"
    return d


def plot_elapsed(arms, rdg=1, wrg=0, n_active: int = 13, figsize=(8.4, 4.6)):
    """
    Total elapsed cycles against descriptor length, at constant data moved.

    The most direct result the campaign has, and the one a table buries: the
    optimum is at neither end of the axis. Log-log, because the span is 4x
    vertically and 2048x horizontally.
    """
    import matplotlib.pyplot as plt
    d = elapsed_comparison(arms, rdg=rdg, wrg=wrg, n_active=n_active)
    fig, ax = plt.subplots(figsize=figsize)
    for k in ARM_ORDER:
        if LABEL[k] not in d.columns:
            continue
        y = d[LABEL[k]]
        ax.plot(y.index, y.values, "-o", color=PALETTE[k], lw=2, ms=7,
                label=LABEL[k], zorder=3)
        lo = y.idxmin()
        ax.plot([lo], [y.min()], "o", color=PALETTE[k], ms=13, mfc="none",
                mew=2, zorder=4)
        # depth 2 and depth 4 share an optimum at the same descriptor length,
        # so the labels must not be placed identically.
        dy = {0: -19, 1: -19, 2: 13}[ARM_ORDER.index(k)]
        ha = {0: "center", 1: "right", 2: "left"}[ARM_ORDER.index(k)]
        ax.annotate(f"{y.min()/ACC_CLK_HZ*1e3:.1f} ms", xy=(lo, y.min()),
                    xytext=(0, dy), textcoords="offset points", ha=ha,
                    fontsize=8.5, color=INK["secondary"])
    _ax(ax, "descriptor length (64-bit beats)",
        "elapsed cycles for the whole transfer", logx=True)
    ax.set_yscale("log")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.set_title(f"Same {n_active*SIZE_BEATS_4X4*8/1e6:.2f} MB moved in every point · "
                 f"rings mark each arm's optimum",
                 fontsize=11, color=INK["primary"], loc="left", pad=10)
    fig.tight_layout()
    return fig


def plot_handoff(arms, arm="multiot_ot4_gate1", rdg=1, wrg=0, n_active=13,
                 figsize=(7.8, 5.6)):
    """
    Why long descriptors are slower: the port fills the path's buffering and then
    waits on one accelerator.

    Two rows, not two y-axes. The upper row is what the memory tile's injection
    port achieves; the lower row is how often the buffer in front of it is full.
    They move in opposite directions, which is the mechanism.
    """
    import matplotlib.pyplot as plt
    o = occupancy(arms[arm])
    s = o[(o.rd == rdg) & (o.wr == wrg) & (o.n_active == n_active)].groupby(
        "burst").mean(numeric_only=True)
    col = PALETTE[arm]
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    port = s.ddr_read_beats / s.mem_cyc
    axes[0].plot(s.index, port, "-o", color=col, lw=2, ms=7, zorder=3)
    axes[0].axhline(1.0, color=INK["muted"], ls=":", lw=1.2)
    axes[0].axhline(0.5, color=INK["muted"], ls="--", lw=1.2)
    axes[0].annotate("1.000  the injection port's own limit", xy=(16384, 1.0),
                     xytext=(0, 5), textcoords="offset points", ha="right",
                     fontsize=8, color=INK["secondary"])
    axes[0].annotate("0.500  what ONE receiving accelerator can absorb",
                     xy=(16384, 0.5), xytext=(0, -13), textcoords="offset points",
                     ha="right", fontsize=8, color=INK["secondary"])
    axes[0].set_ylim(0, 1.18)

    blocked = 100 * s.dma_snd_blocked / s.mem_cyc
    axes[1].plot(s.index, blocked, "-o", color=col, lw=2, ms=7, zorder=3)
    axes[1].set_ylim(-3, 58)
    axes[1].annotate("buffer never fills:\nthe port hands off\nand moves on",
                     xy=(8.6, 8), fontsize=8.5, color=INK["secondary"], ha="left",
                     va="bottom")
    axes[1].annotate("buffer permanently full:\nthe port waits on\none accelerator",
                     xy=(15000, 38), fontsize=8.5, color=INK["secondary"],
                     ha="right", va="top")

    _ax(axes[0], None, "beats delivered per\nmemory-tile cycle", logx=True)
    _ax(axes[1], "descriptor length (64-bit beats)",
        "outbound buffer full\n(% of cycles)", logx=True)
    fig.suptitle(f"{LABEL[arm]} · {n_active} accelerators, read-only",
                 fontsize=11, color=INK["primary"], x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_peak_by_mix(arms, figsize=(7.8, 3.8)):
    """
    Best port utilisation reached, split by traffic direction mix.

    An AXI port cannot see direction, so if the port were the constraint these
    four bars would be equal height. They are not, and that is the argument.
    """
    import matplotlib.pyplot as plt
    g = occupancy_all(arms)
    order = [(1, 0, "read-only\n1:0"), (1, 4, "write-heavy\n1:4"),
             (4, 1, "read-heavy\n4:1"), (1, 1, "balanced\n1:1")]
    vals, labs = [], []
    for rd, wr, lab in order:
        vals.append(100 * g[(g.rd == rd) & (g.wr == wr)].tot_beats_per_cyc.max())
        labs.append(lab)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labs, vals, color=PALETTE["baseline_ot1_gate0"], width=.62,
                  zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}%", xy=(b.get_x() + b.get_width()/2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    color=INK["primary"])
    ax.axhline(100, color=INK["muted"], ls=":", lw=1.2)
    ax.annotate("the MIG AXI port's limit", xy=(3.42, 100), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8,
                color=INK["secondary"])
    _ax(ax, None, "best port utilisation\nreached anywhere (%)")
    ax.set_ylim(0, 112)
    ax.set_title("A port cannot see direction · if it were the limit, "
                 "these would be equal",
                 fontsize=11, color=INK["primary"], loc="left", pad=10)
    fig.tight_layout()
    return fig


def plot_effective_concurrency(arms, rdg=1, wrg=0, n_active=13, figsize=(8.4, 4.4)):
    """
    Aggregate expressed in units of one receiving accelerator's absorption rate.

    Three regimes read directly off the 1.0 line: below it nothing is kept fed,
    at it exactly one accelerator is being drained, above it several are draining
    at once from buffering.
    """
    import matplotlib.pyplot as plt
    d = effective_concurrency(arms, rdg=rdg, wrg=wrg, n_active=n_active)
    fig, ax = plt.subplots(figsize=figsize)
    for k in ARM_ORDER:
        if LABEL[k] not in d.columns:
            continue
        ax.plot(d.index, d[LABEL[k]], "-o", color=PALETTE[k], lw=2, ms=7,
                label=LABEL[k], zorder=3)
    ax.axhline(1.0, color=INK["muted"], ls="--", lw=1.4)
    ax.annotate("1.0  =  exactly one accelerator's worth", xy=(16384, 1.0),
                xytext=(0, 6), textcoords="offset points", ha="right",
                fontsize=8.5, color=INK["secondary"])
    _ax(ax, "descriptor length (64-bit beats)",
        "aggregate, in units of one\naccelerator's absorption rate", logx=True)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.set_title(f"{n_active} accelerators, "
                 f"{'read-only' if wrg == 0 else 'balanced 1:1'}",
                 fontsize=11, color=INK["primary"], loc="left", pad=10)
    fig.tight_layout()
    return fig
