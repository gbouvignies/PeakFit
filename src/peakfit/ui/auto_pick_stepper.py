"""Step-wise GUI controller for automatic peak picking."""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from peakfit.auto_pick.types import AutoPickCycleAction
from peakfit.plot.qt_core import (
    QApplication,
    QComboBox,
    QEventLoop,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from peakfit.plot.widgets.matplotlib_backend import MatplotlibBackend

if TYPE_CHECKING:
    from collections.abc import Callable

    from peakfit.auto_pick.types import AutoPickCycleReport
    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.domain.spectrum import Spectra


_MIN_SPECTRAL_DIMS = 2
_PAIR_LEN = 2
type NavigationCommand = Literal["next_cluster", "previous_cluster"]


def _spectral_axis_limits(spectra: Spectra) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return X/Y ppm limits for plotting (direct dim on X, first indirect on Y)."""
    if spectra.n_spectral_dims < _MIN_SPECTRAL_DIMS:
        msg = (
            "Interactive auto-pick stepping requires at least "
            f"{_MIN_SPECTRAL_DIMS} spectral dimensions."
        )
        raise ValueError(msg)

    x_param = spectra.spectral_params[-1]
    y_param = spectra.spectral_params[0]
    x_limits = (float(x_param.pts2ppm(0.0)), float(x_param.pts2ppm(float(x_param.size - 1))))
    y_limits = (float(y_param.pts2ppm(0.0)), float(y_param.pts2ppm(float(y_param.size - 1))))
    return x_limits, y_limits


def _peaks_to_plot_table(peaks: list[Peak]) -> pd.DataFrame | None:
    """Convert peaks to the DataFrame layout expected by MatplotlibBackend."""
    if not peaks:
        return None

    rows = [
        {
            "name": peak.name,
            "y0_ppm": float(peak.positions[0]),
            "x0_ppm": float(peak.positions[-1]),
        }
        for peak in peaks
    ]
    return pd.DataFrame(rows, columns=["name", "y0_ppm", "x0_ppm"])


def _expand_limits(
    limits: tuple[float, float] | None,
    *,
    fraction: float = 0.1,
    minimum_span: float = 0.02,
) -> tuple[float, float] | None:
    """Return limits expanded by a small margin for visual context."""
    if limits is None:
        return None
    low, high = sorted(limits)
    span = max(high - low, minimum_span)
    margin = span * fraction
    return (low - margin, high + margin)


class _AutoPickStepWindow(QMainWindow):
    """Interactive window to manually steer peak additions within each ROI."""

    def __init__(self, spectra: Spectra, contour_level: float) -> None:
        super().__init__()
        self._x_limits, self._y_limits = _spectral_axis_limits(spectra)
        self._contour_level = float(contour_level)
        self._backend = MatplotlibBackend()
        self._wait_loop: QEventLoop | None = None
        self._stop_requested = False
        self._has_rendered = False
        self._pending_action: AutoPickCycleAction | None = None
        self._navigation_queue: list[NavigationCommand] = []
        self._clicked_candidates: list[tuple[float, float]] = []
        self._suggested_candidate: tuple[float, float] | None = None
        self._suggested_candidate_name: str | None = None
        self._candidate_artists: list[Any] = []
        self._candidate_text_artists: list[Any] = []
        self._trial_artists: list[Any] = []
        self._current_iteration: int | None = None
        self._cluster_states: dict[int, tuple[str, int, float]] = {}

        self.setWindowTitle("PeakFit Auto-Pick Stepper")
        self.setGeometry(120, 120, 1000, 800)
        self._init_ui()
        self._backend.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(self._backend.get_widget())

        self._summary = QLabel("Waiting for first auto-pick cycle...")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        self._diagnostics = QLabel("Diagnostics: n/a")
        self._diagnostics.setWordWrap(True)
        layout.addWidget(self._diagnostics)

        self._add_mode_controls(layout)
        self._add_navigator_controls(layout)
        self._add_queue_controls(layout)
        self._add_action_buttons(layout)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            "Click on spectrum to place candidate, then Add Peak / Remove Last / "
            "Release LWs / Previous Cluster / Next Cluster. "
            "Hotkeys: A add, R remove, L release LWs, [ previous, ] next."
        )

    def _add_mode_controls(self, layout: QVBoxLayout) -> None:
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode"))
        self._mode_selector = QComboBox()
        self._mode_selector.addItems(["Manual", "Semi-auto"])
        self._mode_selector.setCurrentIndex(0)
        mode_layout.addWidget(self._mode_selector)
        mode_layout.addWidget(QLabel("View"))
        self._view_selector = QComboBox()
        self._view_selector.addItems(["Exp + Sim", "Exp only", "Sim only", "Residual"])
        self._view_selector.setCurrentIndex(0)
        mode_layout.addWidget(self._view_selector)
        layout.addLayout(mode_layout)

    def _add_navigator_controls(self, layout: QVBoxLayout) -> None:
        navigator_layout = QHBoxLayout()
        navigator_layout.addWidget(QLabel("Cluster"))
        self._cluster_selector = QComboBox()
        navigator_layout.addWidget(self._cluster_selector)
        self._jump_cluster_button = QPushButton("Go")
        self._jump_cluster_button.clicked.connect(self._on_jump_cluster_clicked)
        navigator_layout.addWidget(self._jump_cluster_button)
        self._save_session_button = QPushButton("Save Session")
        self._load_session_button = QPushButton("Load Session")
        self._save_session_button.clicked.connect(self._on_save_session_clicked)
        self._load_session_button.clicked.connect(self._on_load_session_clicked)
        navigator_layout.addWidget(self._save_session_button)
        navigator_layout.addWidget(self._load_session_button)
        layout.addLayout(navigator_layout)

    def _add_queue_controls(self, layout: QVBoxLayout) -> None:
        queue_layout = QHBoxLayout()
        queue_layout.addWidget(QLabel("Queue"))
        self._queue_selector = QComboBox()
        queue_layout.addWidget(self._queue_selector)
        self._queue_up_button = QPushButton("Up")
        self._queue_down_button = QPushButton("Down")
        self._queue_remove_button = QPushButton("Remove")
        self._queue_clear_button = QPushButton("Clear")
        self._queue_up_button.clicked.connect(self._on_queue_up_clicked)
        self._queue_down_button.clicked.connect(self._on_queue_down_clicked)
        self._queue_remove_button.clicked.connect(self._on_queue_remove_clicked)
        self._queue_clear_button.clicked.connect(self._on_queue_clear_clicked)
        queue_layout.addWidget(self._queue_up_button)
        queue_layout.addWidget(self._queue_down_button)
        queue_layout.addWidget(self._queue_remove_button)
        queue_layout.addWidget(self._queue_clear_button)
        layout.addLayout(queue_layout)

    def _add_action_buttons(self, layout: QVBoxLayout) -> None:
        buttons_layout = QHBoxLayout()
        self._add_button = QPushButton("Add Peak")
        self._remove_button = QPushButton("Remove Last Peak")
        self._release_lw_button = QPushButton("Release LWs")
        self._previous_cluster_button = QPushButton("Previous Cluster")
        self._next_cluster_button = QPushButton("Next Cluster")
        self._stop_button = QPushButton("Stop")
        self._add_button.clicked.connect(self._on_add_clicked)
        self._remove_button.clicked.connect(self._on_remove_clicked)
        self._release_lw_button.clicked.connect(self._on_release_lws_clicked)
        self._previous_cluster_button.clicked.connect(self._on_previous_cluster_clicked)
        self._next_cluster_button.clicked.connect(self._on_next_cluster_clicked)
        self._stop_button.clicked.connect(self._on_stop_clicked)
        buttons_layout.addWidget(self._add_button)
        buttons_layout.addWidget(self._remove_button)
        buttons_layout.addWidget(self._release_lw_button)
        buttons_layout.addWidget(self._previous_cluster_button)
        buttons_layout.addWidget(self._next_cluster_button)
        buttons_layout.addWidget(self._stop_button)
        layout.addLayout(buttons_layout)

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        """Treat closing the window as a stop request."""
        self._stop_requested = True
        self._pending_action = AutoPickCycleAction(command="stop")
        self._quit_wait_loop()
        super().closeEvent(event)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        """Keyboard shortcuts for common stepper actions."""
        key_text = event.text().lower()
        if key_text == "a":
            self._on_add_clicked()
            return
        if key_text == "r":
            self._on_remove_clicked()
            return
        if key_text == "l":
            self._on_release_lws_clicked()
            return
        if key_text == "[":
            self._on_previous_cluster_clicked()
            return
        if key_text == "]":
            self._on_next_cluster_clicked()
            return
        super().keyPressEvent(event)

    def _quit_wait_loop(self) -> None:
        if self._wait_loop is not None:
            self._wait_loop.quit()

    def _on_canvas_click(self, event: Any) -> None:
        """Queue clicked location as a manual candidate (y_ppm, x_ppm)."""
        if event.inaxes != self._backend.ax or event.xdata is None or event.ydata is None:
            return
        self._clicked_candidates.append((float(event.ydata), float(event.xdata)))
        self._refresh_candidate_queue()
        self._draw_candidate_marker()
        self._status.showMessage(
            f"Queued manual candidate #{len(self._clicked_candidates)} at "
            f"y={event.ydata:.4f} ppm, x={event.xdata:.4f} ppm."
        )

    def _on_add_clicked(self) -> None:
        manual_mode = self._mode_selector.currentText() == "Manual"
        if manual_mode and not self._clicked_candidates:
            self._status.showMessage(
                "Manual mode: queue at least one candidate before adding peaks."
            )
            return

        self._navigation_queue.clear()
        self._pending_action = AutoPickCycleAction(
            command="continue",
            candidate_ppm_list=list(self._clicked_candidates) if self._clicked_candidates else None,
            allow_suggested_fallback=not manual_mode,
        )
        self._clicked_candidates.clear()
        self._refresh_candidate_queue()
        self._draw_candidate_marker()
        self._quit_wait_loop()

    def _on_remove_clicked(self) -> None:
        self._navigation_queue.clear()
        self._pending_action = AutoPickCycleAction(command="remove_last_peak")
        self._clicked_candidates.clear()
        self._refresh_candidate_queue()
        self._draw_candidate_marker()
        self._quit_wait_loop()

    def _on_next_cluster_clicked(self) -> None:
        self._navigation_queue.clear()
        self._pending_action = AutoPickCycleAction(command="next_cluster")
        self._clicked_candidates.clear()
        self._refresh_candidate_queue()
        self._draw_candidate_marker()
        self._quit_wait_loop()

    def _on_previous_cluster_clicked(self) -> None:
        self._navigation_queue.clear()
        self._pending_action = AutoPickCycleAction(command="previous_cluster")
        self._clicked_candidates.clear()
        self._refresh_candidate_queue()
        self._draw_candidate_marker()
        self._quit_wait_loop()

    def _on_release_lws_clicked(self) -> None:
        self._navigation_queue.clear()
        self._pending_action = AutoPickCycleAction(command="release_linewidths")
        self._draw_candidate_marker()
        self._quit_wait_loop()

    def _on_jump_cluster_clicked(self) -> None:
        target_data = self._cluster_selector.currentData()
        if target_data is None or self._current_iteration is None:
            self._status.showMessage("No cluster selected for navigation.")
            return

        target_iteration = int(target_data)
        current_iteration = int(self._current_iteration)
        delta = target_iteration - current_iteration
        if delta == 0:
            self._status.showMessage(f"Already at cluster {target_iteration}.")
            return

        command: NavigationCommand = "next_cluster" if delta > 0 else "previous_cluster"
        steps = abs(delta)
        self._navigation_queue = [command for _ in range(max(steps - 1, 0))]
        self._pending_action = AutoPickCycleAction(command=command)
        self._clicked_candidates.clear()
        self._refresh_candidate_queue()
        self._draw_candidate_marker()
        self._status.showMessage(
            f"Jumping to cluster {target_iteration} ({steps} step{'s' if steps > 1 else ''})."
        )
        self._quit_wait_loop()

    def _on_save_session_clicked(self) -> None:
        """Save current stepper UI state to a JSON file."""
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Auto-Pick Session",
            "autopick_stepper_session.json",
            "JSON Files (*.json)",
        )
        if not path_str:
            return

        payload = {
            "mode": self._mode_selector.currentText(),
            "view": self._view_selector.currentText(),
            "queue": self._clicked_candidates,
            "selected_cluster": self._cluster_selector.currentData(),
        }
        Path(path_str).write_text(json.dumps(payload, indent=2))
        self._status.showMessage(f"Session saved to {path_str}")

    def _on_load_session_clicked(self) -> None:
        """Load stepper UI state from a JSON file."""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load Auto-Pick Session",
            "",
            "JSON Files (*.json)",
        )
        if not path_str:
            return

        payload = json.loads(Path(path_str).read_text())
        mode = str(payload.get("mode", "Manual"))
        view = str(payload.get("view", "Exp + Sim"))
        queue = payload.get("queue", [])
        selected_cluster = payload.get("selected_cluster")

        self._set_combo_to_text(self._mode_selector, mode)
        self._set_combo_to_text(self._view_selector, view)
        self._clicked_candidates = [
            (float(candidate[0]), float(candidate[1]))
            for candidate in queue
            if isinstance(candidate, (list, tuple)) and len(candidate) == _PAIR_LEN
        ]
        self._refresh_candidate_queue()
        self._draw_candidate_marker()

        if selected_cluster is not None:
            cluster_index = self._cluster_selector.findData(int(selected_cluster))
            if cluster_index >= 0:
                self._cluster_selector.setCurrentIndex(cluster_index)

        self._status.showMessage(f"Session loaded from {path_str}")

    def _on_stop_clicked(self) -> None:
        self._stop_requested = True
        self._navigation_queue.clear()
        self._pending_action = AutoPickCycleAction(command="stop")
        self._status.showMessage("Stop requested by user.")
        self._quit_wait_loop()

    @staticmethod
    def _set_combo_to_text(combo: QComboBox, text: str) -> None:
        """Set combo selection by text when available."""
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _refresh_candidate_queue(self, preferred_index: int | None = None) -> None:
        """Refresh the editable queue widget from clicked candidates."""
        current_index = self._queue_selector.currentIndex()
        target_index = current_index if preferred_index is None else preferred_index

        self._queue_selector.clear()
        labels = self._queued_candidate_labels(len(self._clicked_candidates))
        for idx, ((y_ppm, x_ppm), label) in enumerate(
            zip(self._clicked_candidates, labels, strict=False),
        ):
            text = f"{label}: y={y_ppm:.4f} x={x_ppm:.4f}"
            self._queue_selector.addItem(text, idx)

        if self._clicked_candidates:
            bounded_index = max(0, min(target_index, len(self._clicked_candidates) - 1))
            self._queue_selector.setCurrentIndex(bounded_index)

    def _on_queue_up_clicked(self) -> None:
        """Move selected queued candidate one position up."""
        index = self._queue_selector.currentIndex()
        if index <= 0 or index >= len(self._clicked_candidates):
            return
        self._clicked_candidates[index - 1], self._clicked_candidates[index] = (
            self._clicked_candidates[index],
            self._clicked_candidates[index - 1],
        )
        self._refresh_candidate_queue(index - 1)
        self._draw_candidate_marker()

    def _on_queue_down_clicked(self) -> None:
        """Move selected queued candidate one position down."""
        index = self._queue_selector.currentIndex()
        if index < 0 or index >= len(self._clicked_candidates) - 1:
            return
        self._clicked_candidates[index], self._clicked_candidates[index + 1] = (
            self._clicked_candidates[index + 1],
            self._clicked_candidates[index],
        )
        self._refresh_candidate_queue(index + 1)
        self._draw_candidate_marker()

    def _on_queue_remove_clicked(self) -> None:
        """Remove selected queued candidate."""
        index = self._queue_selector.currentIndex()
        if index < 0 or index >= len(self._clicked_candidates):
            return
        del self._clicked_candidates[index]
        self._refresh_candidate_queue(index)
        self._draw_candidate_marker()

    def _on_queue_clear_clicked(self) -> None:
        """Clear all queued candidates."""
        self._clicked_candidates.clear()
        self._refresh_candidate_queue()
        self._draw_candidate_marker()

    def update_cycle(self, cycle: AutoPickCycleReport) -> None:
        """Update visualization and summary for one completed cycle."""
        if self._current_iteration != cycle.iteration:
            self._clicked_candidates.clear()
            self._refresh_candidate_queue()
            self._current_iteration = cycle.iteration

        self._refresh_cluster_selector(cycle)
        peaks = cycle.roi_peaks
        peak_table = _peaks_to_plot_table(peaks)
        self._suggested_candidate = cycle.next_candidate_ppm
        self._suggested_candidate_name = cycle.next_candidate_name
        show_spectra = self._show_spectra_flags()
        difference_projection = cycle.experimental_projection - cycle.simulated_projection

        roi_x = _expand_limits(cycle.roi_x_limits, fraction=0.25, minimum_span=0.01)
        roi_y = _expand_limits(cycle.roi_y_limits, fraction=0.25, minimum_span=0.01)
        view_x_limits = list(roi_x) if roi_x is not None else list(self._x_limits)
        view_y_limits = list(roi_y) if roi_y is not None else list(self._y_limits)

        self._backend.plot(
            cycle.experimental_projection,
            cycle.simulated_projection,
            difference_projection,
            peak_table,
            show_spectra,
            contour_level=self._contour_level,
            noise_level=1.0,
            current_plane=0,
            xlim=list(self._x_limits),
            ylim=list(self._y_limits),
            reset_view=True,
        )
        self._backend.ax.set_xlim(*sorted(view_x_limits, reverse=True))
        self._backend.ax.set_ylim(*sorted(view_y_limits, reverse=True))
        self._draw_trial_marker(cycle)
        self._draw_candidate_marker()
        self._has_rendered = True

        accepted_trials = sum(1 for trial in cycle.trials if trial.accepted)
        last_reason = cycle.trials[-1].reason if cycle.trials else "no_trial"
        last_fit_steps = cycle.trials[-1].fit_step_rounds if cycle.trials else 0
        last_cs_bound = cycle.trials[-1].cs_at_constraint if cycle.trials else False
        stage_label = "peak-added" if cycle.stage == "peak_added" else "cycle-complete"
        self._summary.setText(
            f"{stage_label} | cycle {cycle.iteration} | roi={cycle.roi_size} | "
            f"seed={cycle.seed_height:.3e} | "
            f"trials={len(cycle.trials)} accepted_trials={accepted_trials} | "
            f"result={'accepted' if cycle.accepted else 'rejected'} | "
            f"peaks +{cycle.peaks_added} (total {cycle.total_peaks}) | "
            f"residual max={cycle.working_max_after:.3e} | "
            f"last={last_reason} fit_steps={last_fit_steps} "
            f"cs_bound={'yes' if last_cs_bound else 'no'}"
        )
        self._update_diagnostics(cycle)
        prompt = (
            "Choose action: Add Peak, Remove Last Peak, Release LWs, "
            "Previous Cluster, Next Cluster, Jump, or Stop."
        )
        if cycle.feedback_message:
            self._status.showMessage(f"{cycle.feedback_message} {prompt}")
        else:
            self._status.showMessage(prompt)

    def _refresh_cluster_selector(self, cycle: AutoPickCycleReport) -> None:
        """Refresh navigator entries and keep current cluster selected."""
        status = (
            "in progress"
            if cycle.stage == "peak_added"
            else ("accepted" if cycle.accepted else "rejected")
        )
        self._cluster_states[cycle.iteration] = (status, cycle.peaks_added, cycle.seed_height)

        selected_iteration = self._cluster_selector.currentData()
        self._cluster_selector.clear()
        for iteration in sorted(self._cluster_states):
            state, peaks_added, seed_height = self._cluster_states[iteration]
            label = f"#{iteration} {state} peaks={peaks_added} seed={seed_height:.2e}"
            self._cluster_selector.addItem(label, iteration)

        target_iteration = cycle.iteration
        if selected_iteration in self._cluster_states:
            target_iteration = int(selected_iteration)

        index = self._cluster_selector.findData(target_iteration)
        if index >= 0:
            self._cluster_selector.setCurrentIndex(index)

    def _show_spectra_flags(self) -> dict[str, bool]:
        """Return contour visibility flags based on selected compare mode."""
        mode = self._view_selector.currentText()
        if mode == "Exp only":
            return {"spectrum_exp": True, "spectrum_sim": False, "difference": False}
        if mode == "Sim only":
            return {"spectrum_exp": False, "spectrum_sim": True, "difference": False}
        if mode == "Residual":
            return {"spectrum_exp": False, "spectrum_sim": False, "difference": True}
        return {"spectrum_exp": True, "spectrum_sim": True, "difference": False}

    def _update_diagnostics(self, cycle: AutoPickCycleReport) -> None:
        """Render per-step diagnostics in the side text block."""
        mode = self._mode_selector.currentText()
        view = self._view_selector.currentText()
        queue_size = len(self._clicked_candidates)
        if not cycle.trials:
            self._diagnostics.setText(
                f"Diagnostics | mode={mode} view={view} queue={queue_size} | no trial yet"
            )
            return

        trial = cycle.trials[-1]
        if trial.f_test is None:
            f_text = "n/a"
            p_text = "n/a"
            rss_text = "n/a"
        else:
            f_val = trial.f_test.f_stat
            p_val = trial.f_test.p_value
            f_text = "n/a" if f_val is None else f"{f_val:.3e}"
            p_text = "n/a" if p_val is None else f"{p_val:.3e}"
            rss_text = f"{trial.f_test.old_rss:.3e} -> {trial.f_test.new_rss:.3e}"

        self._diagnostics.setText(
            "Diagnostics | "
            f"mode={mode} view={view} queue={queue_size} | "
            f"trial={trial.trial_index} "
            f"decision={'accept' if trial.accepted else 'reject'} "
            f"reason={trial.reason} "
            f"F={f_text} p={p_text} "
            f"RSS={rss_text} "
            f"fit_steps={trial.fit_step_rounds} "
            f"cs_bound={'yes' if trial.cs_at_constraint else 'no'}"
        )

    def _draw_trial_marker(self, cycle: AutoPickCycleReport) -> None:
        """Overlay last-trial marker for quick accept/reject visual feedback."""
        for artist in self._trial_artists:
            self._safe_remove_artist(artist)
        self._trial_artists.clear()

        if not cycle.trials:
            return

        trial = cycle.trials[-1]
        y_ppm = float(trial.candidate_ppm[0])
        x_ppm = float(trial.candidate_ppm[-1])
        marker_style = "o" if trial.accepted else "x"
        color = "tab:orange" if trial.accepted else "tab:red"
        marker = self._backend.ax.scatter(
            [x_ppm],
            [y_ppm],
            marker=marker_style,
            s=40,
            color=color,
            linewidths=1.0,
            zorder=114,
        )
        label = self._backend.ax.annotate(
            "accepted" if trial.accepted else "rejected",
            (x_ppm, y_ppm),
            textcoords="offset points",
            xytext=(6, -10),
            color=color,
            zorder=114,
        )
        self._trial_artists.append(marker)
        self._trial_artists.append(label)

    def _queued_candidate_labels(self, count: int) -> list[str]:
        """Build labels for queued manual candidates from the next peak name."""
        if count == 0:
            return []

        name = self._suggested_candidate_name or ""
        if name.startswith("ap") and name[2:].isdigit():
            start = int(name[2:])
            return [f"ap{start + i}" for i in range(count)]
        return [f"manual{i + 1}" for i in range(count)]

    @staticmethod
    def _safe_remove_artist(artist: Any | None) -> None:
        """Remove a matplotlib artist when possible, ignoring stale handles."""
        if artist is None:
            return
        with suppress(ValueError, NotImplementedError):
            artist.remove()

    def _draw_candidate_marker(self) -> None:
        """Draw queued/manual candidates in gray on top of current contours."""
        for artist in self._candidate_artists:
            self._safe_remove_artist(artist)
        self._candidate_artists.clear()
        for artist in self._candidate_text_artists:
            self._safe_remove_artist(artist)
        self._candidate_text_artists.clear()

        if self._clicked_candidates:
            candidates = list(self._clicked_candidates)
            labels = self._queued_candidate_labels(len(candidates))
        elif self._suggested_candidate is not None:
            candidates = [self._suggested_candidate]
            labels = [self._suggested_candidate_name or "next"]
        else:
            self._backend.canvas.draw_idle()
            return

        for (y_ppm, x_ppm), label in zip(candidates, labels, strict=False):
            marker = self._backend.ax.scatter(
                [x_ppm],
                [y_ppm],
                marker="o",
                s=32,
                color="gray",
                edgecolors="gray",
                linewidths=0.6,
                zorder=115,
            )
            text = self._backend.ax.annotate(
                label,
                (x_ppm, y_ppm),
                textcoords="offset points",
                xytext=(5, 5),
                color="gray",
                zorder=116,
            )
            self._candidate_artists.append(marker)
            self._candidate_text_artists.append(text)
        self._backend.canvas.draw_idle()

    def wait_for_action(self) -> AutoPickCycleAction:
        """Block until user chooses an action for the current ROI state."""
        if self._stop_requested:
            return AutoPickCycleAction(command="stop")

        self._pending_action = None
        self._wait_loop = QEventLoop(self)
        self._wait_loop.exec()
        self._wait_loop = None

        if self._pending_action is None:
            if self._stop_requested:
                return AutoPickCycleAction(command="stop")
            return AutoPickCycleAction(command="continue")
        return self._pending_action

    def consume_navigation_action(self) -> AutoPickCycleAction | None:
        """Consume one queued navigator step (from Jump)."""
        if not self._navigation_queue:
            return None
        command = self._navigation_queue.pop(0)
        return AutoPickCycleAction(command=command)

    def stop_requested(self) -> bool:
        """Return True when the user requested to stop auto-pick."""
        return self._stop_requested


class AutoPickStepController:
    """Controller that plugs GUI step-through into auto-pick callbacks."""

    def __init__(self) -> None:
        app = QApplication.instance()
        self._app = app or QApplication([])
        self._window: _AutoPickStepWindow | None = None
        self._skip_cycle_complete_prompt = False

    def bind(
        self,
        spectra: Spectra,
        contour_level: float,
    ) -> Callable[[AutoPickCycleReport], AutoPickCycleAction]:
        """Create and show the stepper window, then return cycle callback."""
        self._window = _AutoPickStepWindow(spectra=spectra, contour_level=contour_level)
        self._window.show()
        self._app.processEvents()
        return self.on_cycle

    def on_cycle(self, cycle: AutoPickCycleReport) -> AutoPickCycleAction:
        """Handle one cycle update and wait for user action when needed."""
        if self._window is None:
            return AutoPickCycleAction(command="continue")
        self._window.update_cycle(cycle)
        self._app.processEvents()
        if self._window.stop_requested():
            return AutoPickCycleAction(command="stop")
        queued_action = self._window.consume_navigation_action()
        if queued_action is not None:
            return queued_action
        if cycle.stage == "peak_added":
            return self._handle_peak_added_cycle()
        return self._handle_cycle_complete()

    def _handle_peak_added_cycle(self) -> AutoPickCycleAction:
        """Handle user choice after a peak-added update."""
        if self._window is None:
            return AutoPickCycleAction(command="continue")
        action = self._window.wait_for_action()
        self._skip_cycle_complete_prompt = action.command in {
            "next_cluster",
            "previous_cluster",
        }
        return action

    def _handle_cycle_complete(self) -> AutoPickCycleAction:
        """Wait for a cluster-level navigation action."""
        if self._window is None:
            return AutoPickCycleAction(command="continue")
        # Do not proceed to next ROI until user explicitly requests it.
        if self._skip_cycle_complete_prompt:
            self._skip_cycle_complete_prompt = False
            return AutoPickCycleAction(command="continue")

        self._window.statusBar().showMessage(
            "Cluster complete. Click Previous Cluster, Next Cluster, or Stop."
        )
        while True:
            queued_action = self._window.consume_navigation_action()
            if queued_action is not None:
                return queued_action
            action = self._window.wait_for_action()
            if action.command in {"previous_cluster", "next_cluster", "stop"}:
                return action
            self._window.statusBar().showMessage(
                "Cluster complete. Click Previous Cluster, Next Cluster, or Stop."
            )

    def close(self) -> None:
        """Close the stepper window and flush pending UI events."""
        if self._window is not None:
            self._window.close()
            self._window = None
        self._app.processEvents()
