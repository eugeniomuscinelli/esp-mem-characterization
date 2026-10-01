# ESP memory system under DMA load

Where the [ESP](https://esp.cs.columbia.edu/) memory path saturates when accelerators push
DMA traffic at it, and what multiple outstanding transactions (multiOT) at the memory tile
change about it.

Two campaigns on a Xilinx VCU118 (xcvu9p), each with three bitstreams differing only in how
many AXI reads the memory-tile DMA proxy may keep in flight — 1 (stock ESP), 2, and 4.

| report | SoC | accelerators | runs per arm | |
|---|---|---|---|---|
| **3×3** | 3×3 mesh, 9 tiles | 6 | 193 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugeniomuscinelli/esp-mem-characterization/blob/main/characterization_report.ipynb) |
| **4×4** | 4×4 mesh, 16 tiles | 13 | 505 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugeniomuscinelli/esp-mem-characterization/blob/main/characterization_report_4x4.ipynb) |

Both notebooks ship with their outputs, so they render fully without running anything.

## The result

**The limit at long descriptors is one 64-bit beat per accelerator clock cycle** — 625 MB/s,
0.500 beats per memory-tile cycle. Measured at 0.9998–1.0016 words per accelerator cycle in
every arm at every concurrency level, in both campaigns, out to 13 accelerators, with no
fitted parameter. Thirteen accelerators deliver what one delivers.

**That is not the memory system's capacity.** The same path carries 0.8819 beats/cycle —
1102 MB/s, 88% of the MIG AXI port — at 32-beat descriptors. DRAM, the MIG port, AXI
segmentation, refresh and bank conflicts are all excluded by that one measurement.

**multiOT recovers throughput below the limit, not at it**, and the benefit grows with
fan-in: read-only at burst 8 gives 1.53× at 4 accelerators, 2.20× at 6, and **3.10× at 10**,
then plateaus. It gives exactly nothing at descriptors of 4096 beats and above, and exactly
nothing with a single accelerator — the accelerator-side DMA engine is untouched in every arm
and issues one transaction at a time.

**NoC backpressure is a consequence of delivery, not a constraint on it.** This holds for
both directions. The outbound response plane's stalls converge on one per delivered beat; the
inbound request plane sits at 93–96% while throughput over the same points varies 3.4×. Both
correlate *positively* with throughput.

## Why two campaigns

The 3×3 ran first. The 4×4 exists because that campaign left three things unresolved, and in
two cases could not have resolved them:

- It stopped at 6 accelerators, which turned out to be just below where multiOT's benefit
  peaks and exactly at the threshold where inbound backpressure begins. Inbound stall is
  **0.0% at 1, 2, 4 and 6 accelerators** and 93–98% at 10 and 13.
- Its counter for inbound backpressure was wired to a queue this workload never uses, so it
  read zero regardless. That was unmeasured, not absent.
- Its burst axis stepped 32 → 128 → 256, bracketing the throughput peak without locating it.
  The peak is at burst 64 for the baseline and burst 32 for both multiOT arms.

The 4×4 also carries a free-running cycle counter in the monitor's own clock domain, so its
occupancies are exact; the 3×3's starvation figures exceeded 100% in 78 of 576 runs and are
usable only as a trend.

## Contents

| path | what it is |
|---|---|
| `characterization_report.ipynb` | the 3×3 report |
| `characterization_report_4x4.ipynb` | the 4×4 report |
| `charlib.py` | all loading and analysis; `load_all()` for the 3×3, `load_all("4x4")` for the 4×4 |
| `run_baseline_ot1_gate0_bp/`, `run_multiot_ot4_gate1_bp/`, `run_multiot_ot2_gate1_bp/` | 3×3 data |
| `run_4x4_baseline/`, `run_4x4_ot4/`, `run_4x4_ot2/` | 4×4 data |

Each data directory holds `runs.csv` (one row per accelerator per run), `mem.csv` (one row
per memory tile per run) and `tile.csv` (one row per tile per run).

## Running it

**In Colab** — click either badge. The first cell clones this repo into the VM.

**Locally:**

```bash
pip install jupyterlab pandas matplotlib
jupyter lab
```

`charlib.py` resolves the data directories relative to itself, so keep them beside it.
`python3 charlib.py` runs a self-check.

## Reading it honestly

Each report separates what it **establishes** from what it **does not establish** from what
stays **open**, and says which of its own open questions the other campaign closes. Things
neither campaign establishes, stated there rather than glossed:

- **Whether the memory-tile proxy ever drains two destinations at once.** Non-preemptive
  single-descriptor service is the account that fits every observation, but no counter
  observes destination concurrency, so it is inference and is labelled as such.
- **What the posted-write barrier costs.** No barrier-off arm was built, so every multiOT
  number includes it.
- **Which physical structure sets the 0.500 quantum** — the accelerator-tile clock crossing,
  a half-rate datapath in the proxy, or the response plane's flit width. All three predict
  exactly 0.500 because the clock ratio is exactly 2, and neither dataset separates them.

The 4×4 report also records a result that does not flatter the feature under test: multiOT
depth 4 is up to **6.0% slower** than the baseline in balanced traffic at 2 accelerators,
bursts 16–64, reproducibly (repeat spread 0.04%). Read-only shows no such regression anywhere.

## Hardware and software

ESP SoC generator (Columbia SLD), Ariane RV64 CPU, VCU118 / xcvu9p. Memory tile and NoC at
156.25 MHz; accelerator, CPU and IO tiles at 78.125 MHz — the ratio is exactly 2, and it
matters. ESP caches disabled so every DMA beat is real DRAM traffic. The traffic generator is
a Catapult HLS SystemC accelerator that performs no computation.
