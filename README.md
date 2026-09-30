# ESP memory-system characterization under DMA load

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugeniomuscinelli/esp-mem-characterization/blob/main/characterization_report.ipynb)

Measurements of where the [ESP](https://esp.cs.columbia.edu/) memory path saturates when
accelerators push DMA traffic at it, and what multiple outstanding transactions (multiOT) at
the memory tile change about it.

Three FPGA bitstreams on a Xilinx VCU118 (xcvu9p), 3x3 ESP mesh with six DMA-generator
accelerators and one memory tile. The arms differ in how many AXI reads the memory-tile DMA
proxy may keep in flight: **1** (stock ESP), **2**, and **4**. 193 runs per arm sweeping
descriptor granularity, read:write ratio and the number of concurrently active accelerators.

## The result

The binding constraint at long descriptors is **one 64-bit beat per accelerator clock cycle**
— 625 MB/s, or 0.500 beats per memory-tile cycle — made a *global* limit by the memory-tile
proxy serving one DMA descriptor at a time. Measured `words_per_acc_cyc` is 1.0001–1.0041
across every arm and every concurrency level, with no fitted parameter. Six accelerators
share exactly one accelerator's worth of bandwidth.

It is **not** DRAM, the MIG AXI port, AXI segmentation, refresh, bank conflicts, or NoC
backpressure. One measurement rules out that whole family: the same path reaches **0.8545
beats/cycle** — 85% of the MIG AXI port — at 32-beat descriptors, where the proxy switches
destination often.

multiOT does not move that constraint and is not aimed at it. What it recovers is throughput
lost *below* the constraint, in the short-descriptor regime where the path is latency-bound:
**2.26x** in wall-clock time on identical work. It delivers nothing at long descriptors, and
nothing at all with a single accelerator — the accelerator-side DMA engine is untouched and
issues one transaction at a time in every arm.

NoC response-plane backpressure turns out to be a *consequence* of delivery rather than the
constraint on it. Stops per delivered beat converge on 1.000; the study's highest-throughput
point has 5.4x *less* backpressure than its lowest; and backpressure does not move when load
goes from 1 accelerator to 6.

## Contents

| path | what it is |
|---|---|
| `characterization_report.ipynb` | the report: method, validation, results, conclusions, open questions, corrections |
| `charlib.py` | all loading and analysis — every number in the report is computed here, none transcribed |
| `run_*_bp/runs.csv` | one row per accelerator per run: timing and accelerator counters |
| `run_*_bp/mem.csv` | one row per memory tile per run: DRAM-port and backpressure counters |
| `run_*_bp/tile.csv` | one row per tile per run: NoC injection and per-port link activity |

The notebook ships with its outputs, so it renders fully without running anything.

## Running it

**In Colab** — click the badge. The first cell clones this repo into the VM; everything else
follows.

**Locally:**

```bash
pip install jupyterlab pandas matplotlib
jupyter lab characterization_report.ipynb
```

`charlib.py` resolves the data directories relative to itself, so keep them beside it.
`python3 charlib.py` runs a self-check.

## Reading it honestly

The report separates what was **measured** from what is **inferred** from what is **not
known**, and section 7 lists what earlier drafts got wrong — including two conclusions that
were confidently stated and then refuted by their own data. The pattern in those errors is
worth more than the corrections: most came from reading a saturated counter as if it were a
constraint.

Three things in particular are *not* established here, and the report says so rather than
filling the gap with a plausible story:

- **Inbound (request-plane) backpressure.** The counter intended to measure it was wired to
  an unused queue and read zero by construction. Not "the request plane never stalls" — it
  was never observed.
- **What the posted-write barrier costs.** No barrier-off arm was built, so every multiOT
  number includes it.
- **Whether the proxy ever drains two destinations at once** — the load-bearing but
  unmeasured half of the serialization explanation.

## Hardware and software

ESP SoC generator (Columbia SLD), Ariane RV64 CPU, VCU118 / xcvu9p. Memory tile and NoC at
156.25 MHz; accelerator, CPU and IO tiles at 78.125 MHz. ESP caches disabled so every DMA
beat is real DRAM traffic. The traffic generator is a Catapult HLS SystemC accelerator that
performs no computation.
