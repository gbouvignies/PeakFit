# PeakFit CLI - UI Specification

## Design Philosophy: "Fire and Forget"

The previous "Dashboard" design (transient, full-screen updates) is ill-suited for long-running batch processes where the user is likely multitasking. The new design uses a **"Stream + Sticky Footer"** approach.

- **Stream (History)**: A scrolling log of completed events. This provides a permanent record of what happened. If the user looks away and comes back, they can scroll up to see previous errors.
- **Sticky Footer (Status)**: A fixed line at the bottom showing global progress and ETA.
- **Context**: The user cares most about _failures_ and _outliers_. Success is boring.

## Implementation Mapping

- Pre‑fit manifest: [src/peakfit/ui/prefit.py](src/peakfit/ui/prefit.py)
- Stream + sticky footer view: [src/peakfit/ui/views.py](src/peakfit/ui/views.py)

---

## 1. Pre-Fit: The Manifest

**Goal:** Establish trust. Prove the program understands the inputs before starting the expensive loop.

### ASCII Mockup

```text
┏━━ PeakFit v0.1.0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                              ┃
┃  Author: Guillaume Bouvignies                                                ┃
┃                                                                              ┃
┃  Input Manifest                                                              ┃
┃  ──────────────                                                              ┃
┃  • Method:     VARPRO (Variable Projection)                                  ┃
┃  • Contour:    100000.0 (Auto: 5.0 * 20000.0)                                ┃
┃  • Clusters:   42 (Segmentation based on 0.5 contour)                        ┃
┃  • Refine:     2 iterations                                                  ┃
┃  • Workers:    8 (Parallel)                                                  ┃
┃                                                                              ┃
┃  Input Files                                                                 ┃
┃  ───────────                                                                 ┃
┃  • Spectrum:   pseudo3d.ft2 (12 planes, Pseudo-3D)                           ┃
┃  • Peak List:  pseudo3d.list (104 peaks)                                     ┃
┃  • Noise:      2.45e+04 (Estimated)                                          ┃
┃                                                                              ┃
┃  Output                                                                      ┃
┃  ──────                                                                      ┃
┃  • Directory:  Fits_20251216_1000                                            ┃
┃                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
   Press [Enter] to start fitting or [Ctrl+C] to abort...
```

### Data Mapping

- **Author**: Hardcoded credit.
- **Method**: Optimizer name + details.
- **Refine**: Iteration count.
- **Files**: Details (planes, peak count). No "PASS" unless explicit check fails (e.g. permission).

---

## 2. Live Execution: The Stream

**Goal:** A heartbeat. Show that work is happening, but highlight interesting events (failures, warnings).

### Layout

1.  **Main Area (Scrolling)**: One line per completed cluster or status update.
2.  **Footer (Fixed)**: Progress bar and aggregate stats.

### ASCII Mockup

```text
Iteration 1/2                                                                    <-- Status Message (Blue)
[10:00:01]  Cluster 1    ✓ Success   χ²: 1.05   Time: 0.5s
[10:00:02]  Cluster 2    ✓ Success   χ²: 0.98   Time: 0.4s
[10:00:05]  Cluster 5    ✓ Success   χ²: 1.10 -> 1.05   Time: 0.5s               <-- Refinement (Green if improved)
Correcting data with neighbors...                                                <-- Status Message (Dim)
Iteration 2/2
[10:00:10]  Cluster 1    ✓ Success   χ²: 1.05 -> 1.05   Time: 0.1s               <-- Stable (Plain)
...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Progress: [████████████████████████------] 75% (32/42) | ETA: 00:45
Stats:    Success: 30 | Fail: 1 | Warn: 1 | Avg Red. χ²: 1.12
```

### Display Logic

**Smart Coloring (WYSIWYG):**

- **Plain**: If the displayed digits are identical (e.g., `1.05 -> 1.05`), text remains neutral/blue.
- **Bold Green**: If `new < old` and displayed digits differ (`1.10 -> 1.05`).
- **Bold Red**: If `new > old` and displayed digits differ (`1.05 -> 1.10`).
- **Time**: Always dimmed (e.g., `[dim]Time: 0.02s[/]`) to prevent artifact highlighting.

**Status Messages:**

- Iteration headers are bold blue.
- Data correction steps are dimmed.

---

## 3. Post-Fit: The Report

**Goal:** Actionability. What needs manual attention?

### ASCII Mockup

```text
┏━━ Run Complete ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                              ┃
┃  Summary                                                                     ┃
┃  ───────                                                                     ┃
┃  • Total Time:    3m 12s                                                     ┃
┃  • Success:       95.2% (40/42)                                              ┃
┃  • Mean Red. χ²:  1.12                                                       ┃
┃                                                                              ┃
┃  Action Required (2)                                                         ┃
┃  ───────────────────                                                         ┃
┃  (These clusters failed or look suspicious)                                  ┃
┃                                                                              ┃
┃  ID / Peaks             | Issue               | Metric       ┃
┃  ───────────────────────┼─────────────────────┼──────────────┃
┃  4 (12N-H, 15N-H)       | Singular Matrix     | -            ┃
┃  3 (22N-H)              | High Red. χ²        | Red. χ²=5.2  ┃
┃                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
   Results saved to: Fits_20251216_1000
```

### Data Mapping

- **Terminology**: "Red. χ²" used consistently.
- **Action Required Table**: Filter results for:
  1.  `success == False`
  2.  `redchi > 5.0` (High Red. χ²)
  3.  `params_at_bound == True`
- **Metric**: Display the specific value that triggered the flag.
