"""Three columns: navigator, spectrum and inspector, around one editing workflow."""

from collections import Counter
from dataclasses import replace
from threading import Event
from typing import TYPE_CHECKING, cast

from PySide6 import QtCore, QtGui, QtWidgets

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.spectrum import get_dimension_label, get_shape_names
from peakfit.gui import formatting
from peakfit.gui import theme as themes
from peakfit.gui.background import ViewerComputation
from peakfit.gui.dataset_fit import dataset_summary
from peakfit.gui.editor.canvas import SpectrumCanvas
from peakfit.gui.editor.changes import RunChanges, compare_runs
from peakfit.gui.editor.fitting import DatasetFitting
from peakfit.gui.editor.inspector import Inspector, find_cluster
from peakfit.gui.editor.labels import ElidedLabel
from peakfit.gui.editor.lens import ClusterLens
from peakfit.gui.editor.navigator import Navigator
from peakfit.gui.editor.review import HighlightRules, review_clusters
from peakfit.gui.editor.setup import DEFAULT_SOURCE, lineshape_text
from peakfit.gui.editor.unexplained import (
    DEFAULT_THRESHOLD,
    UnexplainedRegion,
    nearest_peak,
    unexplained_regions,
)
from peakfit.gui.fit_display import ModelPlanes, dataset_model_plane, normalized_residual
from peakfit.gui.icons import icon
from peakfit.gui.peaks import PeakEdits
from peakfit.gui.project import Analysis, dataset_header
from peakfit.gui.spectrum import display_position

if TYPE_CHECKING:
    from peakfit.gui.peaks import WorkingPeak
    from peakfit.gui.spectrum import Dataset


_RESIDUAL = 2
_WIDE_TOOLBAR = 1200
_SIDES = (280, 320)  # default navigator and inspector widths at 1280 px
_LENS_INSPECTOR = 340  # the inspector widens while the lens is open
_UNEXPLAINED_ROWS = 8  # regions listed before "and N more"
_REGION_CONTEXT = 24  # a framed region keeps at least this many points around it
# Canvas display preferences an analysis records with its view.
_DISPLAY = (
    "contour_threshold",
    "contour_factor",
    "contour_count",
    "residual_threshold",
    "labels_visible",
    "heavy_strokes",
)


class EditorWorkspace(QtWidgets.QWidget):
    """Own committed scientific edits; navigation and drafts never change revision."""

    changed = QtCore.Signal()
    guidance_changed = QtCore.Signal(str)

    def __init__(self, dataset: Dataset) -> None:  # noqa: PLR0915
        super().__init__()
        self.inputs_loading = False
        self.dataset = dataset
        self.edits = PeakEdits(dataset.peaks)
        self.fitting = DatasetFitting(self)
        # Ticks only while a fit job is busy, to show its elapsed time.
        self._fit_clock = QtCore.QTimer(self)
        self._fit_clock.setInterval(200)
        self._fit_clock.timeout.connect(self._show_run_line)
        peak_sources = tuple(source for source in dataset.sources if source[0] == "peak_list")
        # Keep origins for undoable IDs, but expose only sources of current peaks.
        self._peak_sources = {peak.id: peak_sources for peak in dataset.peaks}
        self.selected_id: str | None = None
        self.rules = HighlightRules()  # review highlight thresholds; display only
        self._reviews: tuple[object, list] = (None, [])
        self._changes: tuple[object, RunChanges | None] = (None, None)
        # Signal not explained on this plane: computed on request, kept for its key only.
        self.unexplained_threshold = DEFAULT_THRESHOLD  # k in σ_stat; display only
        # (key, regions, "pending" or the failure)
        self._unexplained: tuple[object, list[UnexplainedRegion] | str | Exception | None] = (
            None,
            None,
        )
        self._unexplained_revision = 0
        self._unexplained_cancel = Event()
        self._unexplained_worker = ViewerComputation(self)
        self._unexplained_worker.completed.connect(self._unexplained_ready)
        # One cache of total-model planes for the overview, lens and residual search.
        self.model_planes = ModelPlanes()
        self.canvas = SpectrumCanvas(dataset, self.model_planes)
        decimals = (self.canvas.decimals_x, self.canvas.decimals_y)
        self.navigator = Navigator(
            self.canvas.axis_title(self.canvas.x_axis),
            self.canvas.axis_title(self.canvas.y_axis),
            decimals,
        )
        self.panel = Inspector(self.canvas.x_axis, self.canvas.y_axis, decimals)
        self.inspector = self.panel.editor
        self.canvas.resolve_edit = self.commit_fields
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(1)
        self.splitter.addWidget(self.navigator)
        # The spectral area shows the overview canvas or, while open, the cluster lens.
        self.lens = ClusterLens(dataset, self.model_planes)
        self.spectral = QtWidgets.QStackedWidget()
        self.spectral.addWidget(self.canvas)
        self.spectral.addWidget(self.lens)
        self.lens_open = False
        self._sizes_before_lens: list[int] = []
        # The lens cluster by stable peak IDs, the run it came from, and the peaks a
        # regrouping notice is about.
        self._lens_ids: tuple[str, ...] = ()
        self._lens_run: str | None = None
        self._membership_ids: frozenset[str] = frozenset()
        self.splitter.addWidget(self.spectral)
        self.splitter.addWidget(self.panel)
        for index, stretch in enumerate((0, 1, 0)):
            self.splitter.setStretchFactor(index, stretch)
        self.splitter.setSizes([_SIDES[0], 1280 - sum(_SIDES), _SIDES[1]])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_toolbar(dataset))
        layout.addWidget(self.splitter, 1)
        layout.addWidget(self._build_plane_strip(dataset))
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("muted")
        self.status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.status.setAccessibleName("Sampled spectrum values")
        layout.addWidget(self.status)
        self.error = QtWidgets.QLabel()
        self.error.setObjectName("error")
        self.error.setWordWrap(True)
        self.error.setContentsMargins(
            themes.SPACE_M, themes.SPACE_XS, themes.SPACE_M, themes.SPACE_XS
        )
        self.error.hide()
        layout.addWidget(self.error)
        self._build_actions()
        # The fit setup shows on the dataset page, with nothing selected.
        self.panel.analysis_page.installEventFilter(self)
        self.canvas.status_changed.connect(self.status.setText)
        self.canvas.hovered.connect(self.navigator.highlight)
        self.navigator.hovered.connect(lambda identity: self.canvas.hover(identity, from_list=True))
        self.navigator.selected.connect(self.select_peak)
        self.navigator.cluster_chosen.connect(lambda i: self.select_peak(i, cluster=True))
        self.panel.cluster_requested.connect(self.open_cluster)
        self.panel.member_chosen.connect(self.select_peak)
        self.panel.lens_requested.connect(self.open_lens)
        self.navigator.lens_requested.connect(self.open_lens)
        self.lens.close_requested.connect(self.close_lens)
        self.lens.escape_pressed.connect(self._lens_escape)
        self.lens.selected.connect(lambda i: self.select_peak(i, cluster=True))
        self.lens.moved.connect(self.move_peak)
        self.lens.placed.connect(self.place_peak)
        self.lens.draft_changed.connect(self._drag_draft)
        self.lens.resolve_edit = self.commit_fields
        self.panel.change_chosen.connect(lambda i, c: self.select_peak(i, cluster=c))
        self.panel.unexplained_requested.connect(self.find_unexplained)
        self.panel.region_chosen.connect(self.frame_region)
        self.navigator.next_warning.connect(lambda: self.fitting.next_warning())
        self.panel.next_warning.connect(lambda: self.fitting.next_warning())
        self.panel.setup.changed.connect(lambda settings: self.fitting.configure(settings))
        self.panel.setup.regions_toggled.connect(self.canvas.set_regions_visible)
        self.panel.plane_requested.connect(self.set_plane)
        self.canvas.selected.connect(self.select_peak)
        self.canvas.placed.connect(self.place_peak)
        self.canvas.moved.connect(self.move_peak)
        self.canvas.draft_changed.connect(self._drag_draft)
        self.navigator.edit_requested.connect(self.focus_editor)
        self.canvas.edit_requested.connect(self.focus_editor)
        self.inspector.commit_requested.connect(self.commit_fields)
        # Enter on a name hands the keyboard back to the spectrum or lens, so A can
        # place the next peak at once. Coordinate fields keep focus and show the
        # exact committed value.
        self.inspector.name.returnPressed.connect(self._focus_spectral_area)
        self.inspector.deleted.connect(self.delete_peaks)
        self.inspector.fields_changed.connect(self._fields_changed)
        self.fitting.state_changed.connect(self.render_fitting)
        self._refresh()
        self.navigator.setVisible(bool(dataset.peaks))
        self.set_plane(1)

    def _tool(self, name: str, tip: str, *, checkable: bool = False) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setCheckable(checkable)
        button.setToolTip(tip)
        button.setAccessibleName(tip.split(" (")[0])
        button.setIconSize(QtCore.QSize(18, 18))
        button.setFixedSize(28, 28)
        self._icons.append((button, name))
        return button

    def _build_toolbar(self, dataset: Dataset) -> QtWidgets.QWidget:  # noqa: PLR0915
        """One row: identity and run, comparison, canvas tools, settings, Fit."""
        self._icons: list[tuple[QtWidgets.QAbstractButton, str]] = []
        self._settings_text = ""
        toolbar = QtWidgets.QWidget()
        toolbar.setObjectName("toolbar")
        toolbar.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        bar = QtWidgets.QHBoxLayout(toolbar)
        bar.setContentsMargins(themes.SPACE_M, 7, themes.SPACE_M, 7)
        bar.setSpacing(6)
        self.peaks_button = self._tool(
            "panel-left", "Navigator: peaks and clusters", checkable=True
        )
        self.peaks_button.setChecked(bool(dataset.peaks))
        self.peaks_button.clicked.connect(self.toggle_navigator)
        bar.addWidget(self.peaks_button)
        self.title = ElidedLabel(dataset.path.stem)
        self.title.setObjectName("title")
        self.title.setMaximumWidth(160)
        bar.addWidget(self.title)
        self.modified = QtWidgets.QLabel()
        self.modified.setObjectName("muted")
        bar.addWidget(self.modified)
        # The run line: currentness and counts first, so a narrow window elides the tail.
        self.summary = ElidedLabel()
        self.summary.setObjectName("muted")
        # The run summary opens the analysis page at the active run.
        self.summary.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.summary.installEventFilter(self)
        bar.addWidget(self.summary, 1)
        self.warnings = QtWidgets.QLabel()
        self.warnings.setObjectName("warning")
        self.warnings.hide()
        bar.addWidget(self.warnings)
        segment = QtWidgets.QHBoxLayout()
        segment.setSpacing(0)
        self.mode_buttons = []
        for index, name in enumerate(("Experimental", "Overlay", "Residual")):
            button = QtWidgets.QPushButton(name)
            button.setObjectName("segment")
            button.setProperty("edge", ("first", "", "last")[index])
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.setEnabled(index == 0)
            button.clicked.connect(lambda _checked=False, mode=index: self.set_mode(mode))
            self.mode_buttons.append(button)
            segment.addWidget(button)
        bar.addLayout(segment)
        bar.addSpacing(themes.SPACE_XS)
        self.add_button = self._tool("add-peak", "Add peak (A)", checkable=True)
        self.add_button.clicked.connect(self.start_placing)
        self.pin_button = self._tool(
            "pin", "Pin traces at the sampled position (Space)", checkable=True
        )
        self.pin_button.toggled.connect(self.canvas.set_pinned)
        self.canvas.pin_changed.connect(self._pin_changed)
        full = self._tool("full", "Full spectrum (Ctrl+0)")
        full.clicked.connect(self.full_view)
        display = self._tool("sliders", "Display: contours, labels, trace scale")
        display.clicked.connect(self.display_dialog)
        for button in (self.add_button, self.pin_button, full, display):
            bar.addWidget(button)
        bar.addSpacing(themes.SPACE_XS)
        self.settings_button = QtWidgets.QPushButton()
        self.settings_button.setObjectName("chip")
        self.settings_button.setToolTip("Fit settings: lineshapes, clustering and passes")
        self.settings_button.clicked.connect(self.show_fit_setup)
        bar.addWidget(self.settings_button)
        self.cancel_fit = QtWidgets.QPushButton("Cancel")
        self.cancel_fit.clicked.connect(lambda: self.fitting.cancel())
        bar.addWidget(self.cancel_fit)
        self.fit_button = QtWidgets.QPushButton("Fit")
        self.fit_button.setObjectName("primary")
        self.fit_button.setToolTip("Fit the complete dataset (Ctrl+Return)")
        self.fit_button.clicked.connect(lambda: self.fitting.fit())
        bar.addWidget(self.fit_button)
        self.inspector_button = self._tool("panel-right", "Inspector", checkable=True)
        self.inspector_button.setChecked(True)
        self.inspector_button.clicked.connect(self.toggle_inspector)
        bar.addWidget(self.inspector_button)
        return toolbar

    def _build_plane_strip(self, dataset: Dataset) -> QtWidgets.QWidget:
        """Plain plane navigation: index, count, slider and the plane value."""
        strip = QtWidgets.QWidget()
        strip.setObjectName("planeStrip")
        strip.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        planes = QtWidgets.QHBoxLayout(strip)
        planes.setContentsMargins(themes.SPACE_M, 6, themes.SPACE_M, 6)
        planes.setSpacing(themes.SPACE_S)
        self.previous = self._tool("chevron-left", "Previous plane")
        self.next = self._tool("chevron-right", "Next plane")
        self.previous.clicked.connect(lambda: self.set_plane(self.canvas.plane))
        self.next.clicked.connect(lambda: self.set_plane(self.canvas.plane + 2))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setAccessibleName("Plane navigation")
        self.spin = QtWidgets.QSpinBox()
        self.spin.setAccessibleName("One-based plane number")
        self.spin.setKeyboardTracking(False)
        self.spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        n_planes = dataset.spectra.data.shape[0]
        self.spin.setFixedWidth(16 + self.spin.fontMetrics().horizontalAdvance(str(n_planes)) * 2)
        for control in (self.slider, self.spin):
            control.setRange(1, n_planes)
            control.setValue(1)
            control.valueChanged.connect(self.set_plane)
        caption = QtWidgets.QLabel("Plane")
        caption.setObjectName("muted")
        count = QtWidgets.QLabel(f"of {n_planes}")
        count.setObjectName("muted")
        self.plane_caption = QtWidgets.QLabel("Plane value")
        self.plane_caption.setObjectName("muted")
        self.plane_label = QtWidgets.QLabel()
        self.plane_label.setObjectName("value")
        self.plane_label.setMinimumWidth(64)
        for widget in (self.previous, self.next, caption, self.spin, count):
            planes.addWidget(widget)
        planes.addSpacing(themes.SPACE_S)
        planes.addWidget(self.slider, 1)
        planes.addSpacing(themes.SPACE_S)
        planes.addWidget(self.plane_caption)
        planes.addWidget(self.plane_label)
        if n_planes == 1:
            for widget in (self.previous, self.next, caption, self.spin, count, self.slider):
                widget.hide()
        return strip

    def apply_theme(self, theme: themes.Theme) -> None:
        """Retint icons and plots; no scientific state or geometry changes."""
        self.theme = theme
        for button, name in self._icons:
            button.setIcon(icon(name, theme))
        self.canvas.apply_theme(theme)
        self.lens.apply_theme(theme)
        self.navigator.apply_theme(theme)
        self.panel.apply_theme(theme)
        self._update_inspector()

    def update_settings_summary(self) -> None:
        """The working fit settings, always visible beside the Fit button."""
        config = self.fitting.configuration
        try:
            shapes = " × ".join(
                name.upper() for name in get_shape_names(config.fitting, self.dataset.spectra)
            )
        except (ValueError, KeyError, AttributeError):
            shapes = config.fitting.lineshape
        level = config.clustering.contour_level
        threshold, clustering = (
            (f"{level:.3g}", f"clustering threshold {level:.6g} (explicit)")
            if level is not None
            else (
                f"{config.clustering.contour_factor:g}σ",
                f"clustering threshold {config.clustering.contour_factor:g} × noise",
            )
        )
        steps = config.fitting.steps
        passes = f"{len(steps)} steps" if steps else f"{config.fitting.refine_iterations} passes"
        self._settings_text = f"{shapes} · {threshold} · {passes}"
        self._adapt_toolbar()
        self.settings_button.setToolTip(
            f"Fit settings: lineshapes {shapes} · {clustering} · {passes}. "
            "Click to open the fit setup."
        )

    def show_fit_setup(self) -> None:
        """Clear the selection and open the inspector at the fit setup."""
        if not self.commit_fields():
            return
        self.inspector_button.setChecked(True)
        self.toggle_inspector()
        self.select_peak(None)
        self.panel.reveal("setup")
        self.fitting.setup_seen = True

    def show_analysis_page(self) -> None:
        """Clear the selection and open the inspector at the active run's review."""
        if not self.commit_fields():
            return
        self.inspector_button.setChecked(True)
        self.toggle_inspector()
        self.select_peak(None)
        self.panel.reveal("review")

    def _shapes(self) -> list[tuple[str, str, str]]:
        """Resolved lineshape per spectral axis, with where it came from."""
        config = self.fitting.configuration.fitting
        spectra = self.dataset.spectra
        try:
            names = get_shape_names(config, spectra)
        except (ValueError, KeyError, AttributeError) as exc:
            return [("Lineshape", "unresolved", str(exc))]
        rows = []
        for axis, name in zip(spectra.spectral_params, names, strict=True):
            if config.lineshape != "auto":
                source = "explicit"
            elif name == "pvoigt":
                source = DEFAULT_SOURCE
            else:
                source = "from NMRPipe apodization"
            title = formatting.nucleus(axis.nucleus, axis.label)
            rows.append((title, lineshape_text(name), source))
        return rows

    def _fitting_regions(self):
        """The active run's recorded regions, or else the current preparation's."""
        fitting = self.fitting
        result = fitting.result
        if result is not None and fitting.current:
            ids = {i.engine_name: i.working_id for i in result.inputs.identities}
            members = {c.cluster_id: c.peak_names for c in result.outcome.clusters}
            regions = tuple(
                (s.grid_indices, tuple(ids[n] for n in members[s.cluster_id] if n in ids))
                for s in result.completed_fit.samples
            )
            return ("run", result.run_id), regions, f"recorded in run {result.run_id[:8]}"
        prepared = fitting.prepared
        if prepared is not None:
            ids = {i.engine_name: i.working_id for i in prepared.identities}
            regions = tuple(
                (tuple(c.grid_indices), tuple(ids[p.name] for p in c.peaks))
                for c in prepared.clusters
            )
            return ("preview", id(prepared)), regions, "preview for current settings"
        return None, (), ""

    def _adapt_toolbar(self) -> None:
        """Below 1200 px the run line needs the room; tooltips keep the full text."""
        wide = self.width() >= _WIDE_TOOLBAR
        self.title.setMaximumWidth(160 if wide else 120)
        self.settings_button.setText(self._settings_text if wide else "Fit settings")
        count = self.warnings.property("count") or 0
        self.warnings.setText(
            f"▲ {count} cluster{'s' if count != 1 else ''} with warnings" if wide else f"▲ {count}"
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Adapt the toolbar to the window width without changing panel choices."""
        super().resizeEvent(event)
        self._adapt_toolbar()

    def set_inputs_loading(self, loading: bool) -> None:
        """Hold Fit while a window job replaces or writes the scientific inputs."""
        self.inputs_loading = loading
        self._update_actions()

    def set_mode(self, mode: int) -> None:
        """One visible choice controls the plane and both aligned slices."""
        comparison = self.fitting.comparison
        result = comparison.result
        if result is None or not any(c.usable for c in result.outcome.clusters):
            mode = 0
        for index, button in enumerate(self.mode_buttons):
            button.setChecked(index == mode)
        self.canvas.set_comparison(result, mode, label=comparison.label)

    @property
    def peaks(self):
        """Committed immutable input records, independent of the display."""
        return self.edits.peaks

    @property
    def pending(self) -> bool:
        """Uncommitted editor or pointer draft."""
        return (
            self.inspector.pending
            or self.canvas._drag_peak is not None
            or self.lens.dragging is not None
        )

    def analysis_state(self) -> dict[str, object]:
        """What Save records, by the name the Modified tooltip gives each part."""
        fitting = self.fitting
        return {
            "peaks": self.peaks,
            "scientific edits": self.edits.revision,
            "fit settings": fitting.configuration.model_dump_json(),
            "active run": fitting.result.run_id if fitting.result else None,
            "input files": self.dataset.sources,
            "plane values": tuple(self.dataset.spectra.z_values),
            "kept runs": frozenset(fitting.protected),
            "run history": tuple(entry["run_id"] for entry in fitting.journal),
        }

    def show_modified(self, unsaved: tuple[str, ...]) -> None:
        """Show the window's one Modified state: what differs from the saved analysis."""
        self.modified.setText("Modified" if unsaved else "")
        self.modified.setToolTip("Unsaved: " + ", ".join(unsaved) if unsaved else "")

    def capture(self) -> Analysis:
        """The analysis as Save writes it: committed scientific state and modest view state."""
        c = self.canvas
        view = {
            "plane": c.plane,
            "selected_id": self.selected_id,
            "pin": [float(c.ppm_x[c.ix]), float(c.ppm_y[c.iy])],
            "pinned": c.pinned,
            "pin_peak_id": c.pin_peak_id,
            "ranges": c.main.getViewBox().viewRange(),
            "navigator": self.peaks_button.isChecked(),
            "inspector": self.inspector_button.isChecked(),
            "splitter": self.splitter.sizes(),
            "display": {name: getattr(c, name) for name in _DISPLAY},
            "trace_limits": c._trace_limits,
            "mode": next((i for i, b in enumerate(self.mode_buttons) if b.isChecked()), 0),
        }
        fitting = self.fitting
        return Analysis(
            self.dataset,
            dataset_header(self.dataset),
            self.peaks,
            fitting.configuration.model_dump_json(),
            self.edits.revision,
            tuple(fitting.history),
            view,
            journal=tuple(fitting.journal),
            protected=frozenset(fitting.protected),
            active=fitting.completion,
        )

    def restore(self, analysis: Analysis) -> None:
        """Adopt a verified analysis on this workspace's dataset, then its view."""
        self.edits = PeakEdits(analysis.peaks, revision=analysis.revision)
        completion = analysis.completion
        if completion:
            # Only the active result is resident and needs a process-local identity.
            result = completion.result
            snapshot = replace(result.inputs.snapshot, spectrum_id=id(self.dataset.spectra))
            inputs = replace(result.inputs, snapshot=snapshot)
            completion = replace(completion, result=replace(result, inputs=inputs))
        self.fitting.restore(
            PeakFitConfig.model_validate_json(analysis.configuration),
            completion,
            analysis.runs,
            analysis.journal,
            analysis.protected,
        )
        self._refresh()
        v = analysis.view or {}
        c = self.canvas
        for name, value in v.get("display", {}).items():
            if name in _DISPLAY:
                setattr(c, name, value)
        if v.get("trace_limits"):
            c._trace_limits = {int(k): value for k, value in v["trace_limits"].items()}
        c._refresh_labels()
        self.set_plane(v.get("plane", 0) + 1)
        if v.get("selected_id") in {p.id for p in self.peaks}:
            self.select_peak(v["selected_id"])
        if v.get("pin"):
            c.sample(*v["pin"])
            c.set_pinned(v.get("pinned", False))
            c.pin_peak_id = v.get("pin_peak_id")
        if v.get("ranges"):
            x, y = v["ranges"]
            c.main.getViewBox().setRange(xRange=x, yRange=y, padding=0)
        self.peaks_button.setChecked(v.get("navigator", True))
        self.toggle_navigator()
        self.inspector_button.setChecked(v.get("inspector", True))
        self.toggle_inspector()
        # Sizes from another layout (a different number of columns) are not applied.
        if len(v.get("splitter") or ()) == self.splitter.count():
            self.splitter.setSizes(v["splitter"])
        self.set_mode(v.get("mode", 0))
        self.render_fitting()

    def set_plane_values(self, values, records: tuple[tuple[str, str], ...]) -> None:
        """Replace the plane mapping: a scientific edit, never navigation."""
        spectra = self.dataset.spectra.model_copy(update={"z_values": values})
        sources = tuple(s for s in self.dataset.sources if s[0] != "plane_values") + records
        self.dataset = replace(
            self.dataset, spectra=spectra, sources=sources, has_plane_values=True
        )
        self.canvas.dataset = self.dataset
        self.canvas.spectra = spectra
        self.edits.revision += 1
        self.render_fitting()
        self.changed.emit()
        self.set_plane(self.canvas.plane + 1)

    def import_peaks(
        self,
        peaks: tuple[WorkingPeak, ...],
        sources: tuple[tuple[str, str], ...],
        *,
        append: bool,
    ) -> None:
        """Bind fresh imported IDs to their origins in the same undoable handoff."""
        if any(peak.id in self._peak_sources for peak in peaks):
            raise ValueError("Imported peaks require new stable IDs.")
        self.edits.replace_all(self.peaks + peaks if append else peaks)
        self._peak_sources.update((peak.id, sources) for peak in peaks)
        self._refresh()

    def _pin_changed(self, pinned: bool) -> None:
        with QtCore.QSignalBlocker(self.pin_button):
            self.pin_button.setChecked(pinned)
        self.pin_button.setToolTip(
            "Traces pinned · click to follow the pointer (Space)"
            if pinned
            else "Pin traces at the sampled position (Space)"
        )

    def _build_actions(self) -> None:
        self.undo_action = QtGui.QAction("Undo peak edit", self)
        self.redo_action = QtGui.QAction("Redo peak edit", self)
        self.undo_action.setShortcuts(QtGui.QKeySequence.StandardKey.Undo)
        self.redo_action.setShortcuts(QtGui.QKeySequence.StandardKey.Redo)
        self.undo_action.triggered.connect(lambda: self.history(False))
        self.redo_action.triggered.connect(lambda: self.history(True))
        self.add_action = QtGui.QAction("Add peak", self)
        self.add_action.triggered.connect(self.start_placing)
        self.delete_action = QtGui.QAction("Delete selected peaks", self)
        self.delete_action.triggered.connect(self.delete_peaks)
        self.actions_for_menu = [
            self.undo_action,
            self.redo_action,
            self.add_action,
            self.delete_action,
        ]
        for action in self.actions_for_menu:
            self.addAction(action)
        for widget in (self.canvas.main, self.navigator.list):
            for key, callback in (("Delete", self.delete_peaks), ("Backspace", self.delete_peaks)):
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), widget)
                shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(callback)
        for key, callback in (("A", self.start_placing), ("Escape", self.cancel_gesture)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.canvas.main)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
        placing = QtGui.QShortcut(QtGui.QKeySequence("A"), self.lens)
        placing.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        placing.activated.connect(self.start_placing)

    def _fields_changed(self) -> None:
        self._update_actions()
        self.changed.emit()
        if not self.inspector.pending:
            self.error.hide()

    def _drag_draft(self, _draft: bool) -> None:
        self._update_actions()
        self.changed.emit()

    def commit_fields(self) -> bool:
        """Commit a whole valid group or retain its visible invalid text."""
        if not self.inspector.pending:
            return True
        try:
            name, positions = self.inspector.values()
            if self.selected_id is None:
                return True
            self.edits.update(self.selected_id, name, positions)
        except ValueError as exc:
            self.inspector.details.setText(str(exc))
            self.error.setText(str(exc))
            self.error.show()
            return False
        self.error.hide()
        self._refresh()
        return True

    def select_peak(self, identity: str | None, *, cluster: bool = False) -> None:
        """Resolve drafts before changing the one stable inspector identity.

        ``cluster`` opens the inspector on the peak's recorded cluster rather than
        the peak itself; the selected peak stays the unit of editing either way.
        """
        if not self.commit_fields():
            # Restore list identity without overwriting the invalid visible draft.
            with QtCore.QSignalBlocker(self.navigator.list):
                item = self.navigator._items.get(self.selected_id or "")
                if item:
                    self.navigator.list.setCurrentItem(item)
            return
        self.selected_id = identity
        self.canvas.select(identity)
        self._sync_inspector(cluster=cluster)
        self._update_actions()

    def open_cluster(self) -> None:
        """Show the selected peak's recorded cluster, keeping the peak selected."""
        if self.commit_fields():
            self._sync_inspector(cluster=True)

    def _sync_inspector(self, *, cluster: bool | None = None) -> None:
        peak = next((p for p in self.peaks if p.id == self.selected_id), None)
        if cluster is None:
            cluster = self.panel.cluster_view
        result = self.fitting.result
        cluster = cluster and bool(result and peak and find_cluster(result, peak.id))
        self.navigator.select(peak)
        self.panel.set_peak(peak, self.canvas.display(peak) if peak else None, cluster=cluster)
        self._update_inspector()

    def _cluster_members(self, result) -> tuple[str, ...]:
        """Working IDs that the active run fitted together with the selected peak."""
        found = find_cluster(result, self.selected_id) if result and self.selected_id else None
        if found is None:
            return ()
        ids = {i.engine_name: i.working_id for i in result.inputs.identities}
        return tuple(ids[name] for name in found[1].peak_names if name in ids)

    def reviews(self) -> list:
        """The active run's cluster review at the current highlight thresholds."""
        result = self.fitting.result
        key = (result.run_id, self.rules) if result is not None else None
        if key != self._reviews[0]:
            reviews = []
            if result is not None:
                axes = list(self.dataset.spectra.spectral_params)
                titles = [formatting.nucleus(a.nucleus, a.label) for a in axes]
                reviews = review_clusters(result, axes, titles, self.rules)
            self._reviews = (key, reviews)
        return self._reviews[1]

    def _starting_values(self) -> tuple[list[tuple[str, str, str, str]] | None, str]:
        """The selected peak's parameters as the current preparation would submit them."""
        fitting = self.fitting
        prepared = fitting.prepared
        if prepared is None:
            return None, fitting.blocked or "Preparing the current settings…"
        engine = next(
            (i.engine_name for i in prepared.identities if i.working_id == self.selected_id), None
        )
        axes = {
            get_dimension_label(index): (
                formatting.nucleus(a.nucleus, a.label),
                formatting.ppm_decimals(a),
            )
            for index, a in enumerate(self.dataset.spectra.spectral_params)
        }
        rows = []
        # Horizontal (direct) axis first, as in Fitted; then position before width.
        order = list(reversed(axes))
        kinds = ["cs", "lw"]
        parameters = sorted(
            (
                p
                for p in prepared.parameters.values()
                if p.param_id is not None and p.param_id.peak_name == engine
            ),
            key=lambda p: (
                order.index(p.param_id.axis) if p.param_id.axis in order else len(order),
                kinds.index(p.param_id.label) if p.param_id.label in kinds else len(kinds),
            ),
        )
        for parameter in parameters:
            identifier = parameter.param_id
            assert identifier is not None
            title, decimals = axes.get(identifier.axis or "", (identifier.axis or "", 4))
            kind = {"cs": "δ", "lw": "LW"}.get(identifier.label, identifier.label)
            places = decimals if parameter.unit == "ppm" else 2
            low, high = (formatting.number(v, places) for v in (parameter.min, parameter.max))
            freedom = "varied" if parameter.vary else "fixed"
            rows.append(
                (
                    f"{kind} {title}",
                    formatting.number(parameter.value, places),
                    f"range {low} – {high} · {freedom}",
                    parameter.unit,
                )
            )
        return rows, "For the next fit, from the working position and fit setup."

    def _run_changes(self) -> tuple[str, list[tuple]]:
        """What changed since the run this one was refitted from, as page entries."""
        fitting = self.fitting
        result = fitting.result
        parent_id, parent = fitting.parent_result()
        if result is None:
            return "", []
        if parent_id is None:
            return "", [("note", "First run: there is no earlier run to compare with.")]
        title = f"CHANGES FROM RUN {parent_id[:8].upper()}"
        if parent is None:
            return title, [("note", f"Reading run {parent_id[:8]}…")]
        if isinstance(parent, str):
            return title, [("note", f"Run {parent_id[:8]} cannot be compared: {parent}.")]
        axes = list(self.dataset.spectra.spectral_params)
        titles = [formatting.nucleus(a.nucleus, a.label) for a in axes]
        key = (result.run_id, parent_id, self.rules)
        if key != self._changes[0]:
            parent_reviews = review_clusters(parent, axes, titles, self.rules)
            changes = compare_runs(result, self.reviews(), parent, parent_reviews, axes, titles)
            self._changes = (key, changes)
        changes = self._changes[1]
        assert changes is not None
        return title, _change_entries(
            changes, dict(zip(titles, map(formatting.ppm_decimals, axes), strict=True))
        )

    def set_rules(self, rules: HighlightRules) -> None:
        """Change highlight thresholds; no scientific data or run state changes."""
        self.rules = rules
        self.navigator.rules = rules
        self.render_fitting()

    def _update_inspector(self) -> None:
        """Follow the accepted selection and active run; never resolves drafts."""
        fitting = self.fitting
        spectra = self.dataset.spectra
        result = fitting.result
        reviews = self.reviews()
        self.navigator.set_result(result, reviews, current=bool(fitting.current))
        self.canvas.set_cluster(self._cluster_members(result))
        if self.lens_open:
            self._update_lens()
        self.canvas.set_regions(*self._fitting_regions())
        self.panel.setup.show_state(
            fitting.configuration, self._shapes(), fitting.prepared, fitting.blocked
        )
        plane_values = (
            tuple(float(v) for v in spectra.z_values) if self.dataset.has_plane_values else None
        )
        review = next((r for r in reviews if self.selected_id in r.ids), None)
        current, plane = bool(fitting.current), self.canvas.plane
        # Only the page on show fills itself; the other returns at once.
        retained = dict(fitting.completion.retained) if fitting.completion else {}
        self.panel.show_result(
            result,
            current=current,
            busy=fitting.busy,
            plane_values=plane_values,
            plane=plane,
            review=review,
            retained=retained.get(self.selected_id or ""),
        )
        unadopted = self._unadopted()
        self.canvas.set_unadopted(unadopted)
        self.lens.set_unadopted(unadopted)
        self.panel.show_cluster_result(
            result,
            review,
            current=current,
            busy=fitting.busy,
            plane_values=plane_values,
            plane=plane,
        )
        if self.selected_id is not None:
            self.panel.show_starts(*self._starting_values())
        x, y = (
            formatting.nucleus(a.nucleus, a.label) for a in (self.canvas.x_axis, self.canvas.y_axis)
        )
        planes = spectra.data.shape[0]
        if fitting.busy:
            run = fitting.stage
        elif result is not None:
            flagged = self.warnings.property("count") or 0
            run = (
                f"Run {result.run_id[:8]} · {'Current' if fitting.current else 'Outdated'}\n"
                f"{dataset_summary(result, warnings=False)}\n"
                f"Fitted in {result.execution_seconds:.1f} s · {result.completed_at[:19]} UTC"
                + (f"\n▲ {flagged} clusters with numerical warnings" if flagged else "")
            )
        else:
            run = "Not fitted yet. Fit uses every working peak across all planes."
        if fitting.failure:
            run += f"\n{fitting.attempt}: {fitting.failure}"
        if fitting.recovery_error:
            run += "\n▲ " + _recovery_note(fitting.recovery_error)
        self.panel.show_analysis(
            self.dataset.path.stem,
            f"{len(self.peaks)} working peaks · {planes} planes · {x} × {y}",
            run,
            "Select a peak or cluster to see its position, profile, fitted values and "
            "review. Save analysis keeps working state and the latest ten runs; "
            "spectra stay external.",
            fitted=result is not None,
        )
        if self.selected_id is None:
            self.panel.show_changes(*self._run_changes())
            self._show_unexplained()

    def _refresh(self) -> None:
        peak_sources = tuple(
            dict.fromkeys(
                source for peak in self.peaks for source in self._peak_sources.get(peak.id, ())
            )
        )
        sources = tuple(source for source in self.dataset.sources if source[0] != "peak_list")
        sources += peak_sources
        if sources != self.dataset.sources:
            self.dataset = replace(self.dataset, sources=sources)
        self.canvas.dataset = self.dataset
        if self.selected_id not in {p.id for p in self.peaks}:
            self.selected_id = None
        self.canvas.selected_id = self.selected_id
        self.canvas.markers.selected_id = self.selected_id
        self.canvas.set_peaks(self.peaks)
        self.navigator.refresh(self.peaks, self.selected_id)
        self._sync_inspector()
        self.fitting.request_preview()
        self.render_fitting()
        self.changed.emit()

    def render_fitting(self) -> None:
        """Show the fitting state: the lock, comparison, run line and inspector."""
        fitting = self.fitting
        if fitting.busy != self._fit_clock.isActive():
            if fitting.busy:
                self._fit_clock.start()
            else:
                self._fit_clock.stop()
        self._update_actions()
        comparison = fitting.comparison
        usable = comparison.result is not None and any(
            c.usable for c in comparison.result.outcome.clusters
        )
        lens = self.lens_open
        for button in self.mode_buttons:
            button.setEnabled(not lens and (usable or button is self.mode_buttons[0]))
            button.setToolTip(
                "The lens shows all three views; return to the spectrum to choose one"
                if lens
                else f"Compares with {comparison.label}"
                if usable and comparison.historical and button is not self.mode_buttons[0]
                else ""
                if usable or button is self.mode_buttons[0]
                else f"Unavailable: {comparison.reason}"
                if comparison.reason
                else "Requires a run with usable clusters"
            )
        self.pin_button.setEnabled(not lens)
        canvas = self.canvas
        if canvas.comparison_result is not None and (
            comparison.result is not canvas.comparison_result
            or comparison.historical != canvas.historical
        ):
            # Withdraw an incompatible model, or relabel the same run as historical.
            self.set_mode(canvas.comparison_mode if usable else 0)
            if not usable and comparison.reason:
                self.status.setText(f"Comparison withdrawn: {comparison.reason}")
        canvas.set_changes(comparison.added, comparison.departed)
        self._show_run_line()
        flagged = sum(bool(r.warnings) for r in self.reviews()) if not fitting.busy else 0
        self.warnings.setProperty("count", flagged)
        self.warnings.setToolTip(
            f"{flagged} clusters with a numerical warning (unusable, not converged, "
            "uncertainty unreliable or at bound); N in the cluster list moves between them"
        )
        self.warnings.setVisible(bool(flagged))
        self.update_settings_summary()
        self._update_inspector()

    def _show_run_line(self) -> None:
        """One compact status line; inspection and navigation stay available."""
        fitting = self.fitting
        result = fitting.result
        if fitting.busy:
            text = f"{fitting.stage} · {fitting.elapsed:.1f} s · Navigation available"
        elif result is not None:
            # Currentness first: a narrow toolbar elides only the tail.
            text = (
                f"Run {result.run_id[:8]} · {'Current' if fitting.current else 'Outdated'} · "
                + ("▲ Not in recovery copy · " if fitting.recovery_error else "")
                + (fitting.attempt + " · " if fitting.attempt else "")
                + dataset_summary(result, warnings=False)
            )
        else:
            planes = self.dataset.spectra.data.shape[0]
            text = fitting.attempt or f"Not fitted · {len(self.peaks)} peaks · {planes} planes"
        self.summary.setText(text)
        self.summary.setToolTip(
            text
            + ("\n" + fitting.failure if fitting.failure else "")
            + ("\n" + _recovery_note(fitting.recovery_error) if fitting.recovery_error else "")
            + "\nClick to open the analysis page at the active run"
        )

    def _update_actions(self) -> None:
        """Every scientific edit and Fit follow one busy state: the editing lock."""
        fitting = self.fitting
        busy = fitting.busy
        self.canvas.editing_enabled = not busy
        self.lens.editing_enabled = not busy
        self.inspector.setEnabled(not busy)
        self.panel.setup.setEnabled(not busy)
        self.add_button.setEnabled(not busy)
        self.add_action.setEnabled(not busy)
        self.undo_action.setText("Undo " + self.edits.undo_label)
        self.redo_action.setText("Redo " + self.edits.redo_label)
        self.undo_action.setEnabled(self.edits.can_undo and not self.pending and not busy)
        self.redo_action.setEnabled(self.edits.can_redo and not self.pending and not busy)
        self.delete_action.setEnabled(self.selected_id is not None and not busy)
        blocked = fitting.blocked
        self.fit_button.setEnabled(
            bool(self.peaks) and not busy and not self.inputs_loading and not blocked
        )
        self.fit_button.setToolTip(
            f"Fit unavailable: {blocked}" if blocked else "Fit the complete dataset (Ctrl+Return)"
        )
        # While a fit runs its settings cannot change; the run line needs the room.
        self.settings_button.setVisible(not busy)
        self.cancel_fit.setVisible(busy)

    def history(self, redo: bool) -> None:
        """Undo exact records; monotonic scientific revision never recertifies a run."""
        if self.pending or self.fitting.busy:
            return
        self.canvas.cancel_gesture()
        if redo:
            self.edits.redo()
        else:
            self.edits.undo()
        self._refresh()

    def start_placing(self) -> None:
        """Arm one precise placement after resolving any complete text group."""
        if self.fitting.busy:
            return
        if not self.commit_fields():
            self.add_button.setChecked(False)
            return
        self.add_button.setChecked(True)
        if self.lens_open:
            self.lens.set_placing(True)
            self.lens.setFocus()
        else:
            self.canvas.set_placing(True)
            self.canvas.main.setFocus()
        self.status.setText("Place peak · click exact ppm position · Escape cancels")

    def place_peak(self, positions) -> None:
        """Store continuous unfolded ppm from a single click, without snapping."""
        if self.fitting.busy:
            return
        if not self.commit_fields():
            return
        peak = self.edits.add(positions)
        self.cancel_gesture()
        self.selected_id = peak.id
        self._refresh()
        self.canvas.select(peak.id)

    def move_peak(self, identity, positions) -> None:
        """One drag release is one complete precision-preserving input change."""
        if self.fitting.busy:
            return
        if not self.commit_fields():
            self.cancel_gesture()
            return
        peak = next(p for p in self.peaks if p.id == identity)
        self.edits.update(identity, peak.name, positions)
        self._refresh()
        self.canvas.select(identity)

    def delete_peaks(self) -> None:
        """Deletion owns the selected draft; multi-delete is list-only."""
        if self.fitting.busy:
            return
        identities = {self.selected_id} if self.selected_id else set()
        if self.navigator.list.hasFocus():
            identities = {item.data(256) for item in self.navigator.list.selectedItems()}
        if identities:
            self.cancel_gesture()
            self.edits.delete_many(identities)
            self.selected_id = None
            self._refresh()

    def cancel_gesture(self) -> None:
        """Escape on the canvas cancels its current gesture, never numerical work."""
        self.canvas.cancel_gesture()
        self.lens.cancel_gesture()
        self.add_button.setChecked(False)

    def toggle_navigator(self) -> None:
        """Hide or show the navigator; panel choices never change on resize."""
        self.navigator.setVisible(self.peaks_button.isChecked())

    def toggle_inspector(self) -> None:
        """Resolve a draft on explicit hide, since the editor lives in the inspector."""
        visible = self.inspector_button.isChecked()
        if not visible and not self.commit_fields():
            self.inspector_button.setChecked(True)
            return
        self.panel.setVisible(visible)

    def focus_editor(self) -> None:
        """Explicit editing opens the inspector without changing selection."""
        self.inspector_button.setChecked(True)
        self.toggle_inspector()
        if self.selected_id:
            self.inspector.x_field.setFocus()
            self.inspector.x_field.selectAll()

    def set_plane(self, one_based: int) -> None:
        """Plane navigation stays available even with an invalid coordinate draft."""
        plane = max(1, min(one_based, self.dataset.spectra.data.shape[0]))
        for control in (self.slider, self.spin):
            with QtCore.QSignalBlocker(control):
                control.setValue(plane)
        self.canvas.set_plane(plane - 1)
        count = self.dataset.spectra.data.shape[0]
        self.previous.setEnabled(plane > 1)
        self.next.setEnabled(plane < count)
        values = self.dataset.has_plane_values and count > 1
        self.plane_caption.setVisible(values)
        self.plane_label.setText(
            "Single plane"
            if count == 1
            else formatting.intensity(float(self.dataset.spectra.z_values[plane - 1]))
            if values
            else ""
        )
        self.panel.set_plane(plane - 1)
        if self.selected_id is None:
            self._show_unexplained()
        if self.lens_open:
            self._update_lens()

    def open_lens(self, identity: str | None = None) -> None:
        """Replace the spectral area with the lens on the selected peak's recorded cluster."""
        if identity is not None:
            self.select_peak(identity, cluster=True)
        elif not self.commit_fields():
            return
        result = self.fitting.result
        if result is None or not self.selected_id or find_cluster(result, self.selected_id) is None:
            return
        if not self.lens_open:
            self.lens_open = True
            self._sizes_before_lens = self.splitter.sizes()
            sizes = list(self._sizes_before_lens)
            if sizes[2] < _LENS_INSPECTOR:  # the inspector widens for members side by side
                sizes[1] -= _LENS_INSPECTOR - sizes[2]
                sizes[2] = _LENS_INSPECTOR
                self.splitter.setSizes(sizes)
            self.spectral.setCurrentWidget(self.lens)
            self.panel.cluster_page.lens_button.hide()
        self._sync_inspector(cluster=True)
        self.render_fitting()
        self.lens.setFocus()
        self.guidance_changed.emit(self.guidance())

    def _unexplained_key(self):
        result = self.fitting.comparison.result
        if result is None:
            return None
        return (result.run_id, self.canvas.plane, self.unexplained_threshold)

    def find_unexplained(self) -> None:
        """Compute the current plane's residual regions off Qt; read-only."""
        key = self._unexplained_key()
        if key is None or key == self._unexplained[0]:
            return
        result, plane, threshold = self.fitting.comparison.result, self.canvas.plane, key[2]
        assert result is not None
        observed = self.dataset.spectra.data[plane]
        axes = [axis.model_copy(deep=True) for axis in self.dataset.spectra.spectral_params]
        cached = self.model_planes.get(result.run_id, plane)
        peaks = self._native_peaks()

        self._unexplained_cancel.set()
        self._unexplained_cancel = cancel = Event()

        def compute():
            model = cached or dataset_model_plane(result, axes, plane, cancel=cancel)
            residual = normalized_residual(observed, model.values, result.outcome.noise)
            computed = None if cached else (result.run_id, plane, model)
            return unexplained_regions(residual, threshold, peaks), computed

        self._unexplained = (key, "pending")
        self._unexplained_revision += 1
        self._unexplained_worker.submit("unexplained", self._unexplained_revision, compute)
        self._show_unexplained()

    def _native_peaks(self) -> list[tuple[str, float, float]]:
        """Working peaks as (ID, row, column) in fractional native samples."""
        y_axis, x_axis = self.canvas.y_axis, self.canvas.x_axis
        return [
            (
                peak.id,
                float(y_axis.ppm2pts(display_position(y_axis, peak.positions[0]))),
                float(x_axis.ppm2pts(display_position(x_axis, peak.positions[1]))),
            )
            for peak in self.peaks
        ]

    def _unexplained_ready(self, _kind: str, revision: int, outcome: object) -> None:
        if revision != self._unexplained_revision:
            return
        if not isinstance(outcome, Exception):
            outcome, computed = cast("tuple[list[UnexplainedRegion], tuple | None]", outcome)
            if computed is not None:
                self.model_planes.put(*computed)
        self._unexplained = (self._unexplained[0], outcome)
        self._show_unexplained()
        self.panel.reveal_unexplained()

    def _show_unexplained(self) -> None:
        """Fill the analysis page's list for the current plane, run and threshold."""
        fitting, plane = self.fitting, self.canvas.plane
        k = formatting.number(self.unexplained_threshold, 1).removesuffix(".0")
        note = (
            f"Connected regions of the whole-spectrum residual beyond ±{k} σ_stat. "
            "Signal may be left unmodelled on purpose; this list never proposes peaks "
            "or flags clusters."
        )
        button = f"Find on plane {plane + 1}"
        key = self._unexplained_key()
        if fitting.result is None or fitting.busy:
            self.panel.show_unexplained(button, False, "", [], "Available after a fit. " + note)
            return
        comparison = fitting.comparison
        if key is None:
            self.panel.show_unexplained(
                button, False, f"Unavailable: {comparison.reason}", [], note
            )
            return
        if comparison.historical:
            # The residual is the run's; distances are to today's working peaks.
            note = (
                f"Residual of {comparison.label}. Distances are to the current working "
                "peaks; ◇ marks a peak added since that run. " + note
            )
        found_key, outcome = self._unexplained
        if found_key != key:
            self.panel.show_unexplained(button, True, "", [], note)
            return
        if isinstance(outcome, str):
            self.panel.show_unexplained(button, False, f"Computing plane {plane + 1}…", [], note)
            return
        if isinstance(outcome, Exception):
            self.panel.show_unexplained(button, True, f"Unavailable: {outcome}", [], note)
            return
        regions = outcome if isinstance(outcome, list) else []
        count = len(regions)
        title = (
            f"Plane {plane + 1} · "
            + (f"{count} region{'s' if count != 1 else ''}" if count else "no region")
            + f" beyond ±{k} σ_stat"
            + (f" · {comparison.label}" if comparison.historical else "")
        )
        if comparison.historical:
            # The run's regions stay; nearest peaks follow the edits since that run.
            peaks = self._native_peaks()
            regions = [
                replace(region, nearest=nearest, distance=distance)
                for region in regions
                for nearest, distance in (nearest_peak(region.row, region.column, peaks),)
            ]
        names = {peak.id: peak.name for peak in self.peaks}
        canvas = self.canvas
        x, y = (formatting.nucleus(a.nucleus, a.label) for a in (canvas.x_axis, canvas.y_axis))
        rows = []
        for region in regions[:_UNEXPLAINED_ROWS]:
            location = (
                f"{x} {formatting.number(float(canvas.ppm_x[region.column]), canvas.decimals_x)}"
                f" · {y} {formatting.number(float(canvas.ppm_y[region.row]), canvas.decimals_y)}"
            )
            nearest = (
                f" · {region.distance:.1f} pts from {names.get(region.nearest, '')}"
                + (" ◇" if region.nearest in comparison.added else "")
                if region.nearest is not None
                else ""
            )
            rows.append(
                (
                    location,
                    f"{region.extreme:+.1f} σ · {region.size} sample"
                    f"{'s' if region.size != 1 else ''}{nearest}",
                )
            )
        if count > _UNEXPLAINED_ROWS:
            rows.append(("", f"… and {count - _UNEXPLAINED_ROWS} more"))
        self.panel.show_unexplained(button, False, title, rows, note)

    def frame_region(self, index: int) -> None:
        """Frame a listed region in the overview and sample through its extreme."""
        outcome = self._unexplained[1]
        if not isinstance(outcome, list) or index >= len(outcome):
            return
        region = outcome[index]
        self.canvas.frame(region.rows, region.columns, _REGION_CONTEXT)
        self.canvas.sample(
            float(self.canvas.ppm_x[region.column]), float(self.canvas.ppm_y[region.row])
        )

    def guidance(self) -> str:
        """The status-bar hint for what the spectral area currently allows."""
        if self.lens_open:
            return "Lens · drag selected peak · A adds a peak · Esc exits"
        return "Drag a selected peak to move it · Space pins traces"

    def _focus_spectral_area(self) -> None:
        if not self.inspector.pending:
            (self.lens if self.lens_open else self.canvas.main).setFocus()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """A click on the run summary opens the analysis page; showing it shows the setup."""
        if watched is self.panel.analysis_page and event.type() == QtCore.QEvent.Type.Show:
            self.fitting.setup_seen = True
        if (
            watched is self.summary
            and event.type() == QtCore.QEvent.Type.MouseButtonRelease
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            self.show_analysis_page()
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Escape leaves the lens from any column unless a draft or gesture is active."""
        if event.key() == QtCore.Qt.Key.Key_Escape and self.lens.placing:
            self.cancel_gesture()
            return
        if (
            event.key() == QtCore.Qt.Key.Key_Escape
            and self.lens_open
            and not self.pending
            and QtWidgets.QApplication.activeModalWidget() is None
        ):
            self.close_lens()
            return
        super().keyPressEvent(event)

    def _unadopted(self) -> list[tuple[float, float]]:
        """Display positions of recorded fitted coordinates the active run did not adopt.

        Only peaks retained because of near-collinear fitted shapes: their fitted
        values are valid numbers that the working markers deliberately do not follow.
        """
        completion = self.fitting.completion
        if completion is None:
            return []
        reasons = dict(completion.retained)
        result = completion.result
        engine = {i.working_id: i.engine_name for i in result.inputs.identities}
        values = {
            p.name: p.value for c in result.outcome.clusters for p in c.final_nonlinear_parameters
        }
        x_axis, y_axis = self.canvas.x_axis, self.canvas.y_axis
        x_label, y_label = get_dimension_label(1), get_dimension_label(0)
        points = []
        for identity, reason in reasons.items():
            name = engine.get(identity)
            x = values.get(f"{name}.{x_label}.cs")
            y = values.get(f"{name}.{y_label}.cs")
            if reason.startswith("near-collinear") and x is not None and y is not None:
                points.append((display_position(x_axis, x), display_position(y_axis, y)))
        return points

    def _lens_escape(self) -> None:
        """Escape in the lens cancels a gesture first, then returns to the spectrum."""
        if self.lens.placing or self.lens.dragging is not None:
            self.cancel_gesture()
        else:
            self.close_lens()

    def full_view(self) -> None:
        """Full spectrum in the overview; the cluster's default region in the lens."""
        if self.lens_open:
            self.lens.reframe()
        else:
            self.canvas.full_spectrum()

    def close_lens(self) -> None:
        """Return to the overview with its viewport, mode and panel widths unchanged."""
        if not self.lens_open:
            return
        self.cancel_gesture()
        self.lens_open = False
        self._lens_ids, self._lens_run = (), None
        self._membership_ids = frozenset()
        self.lens.set_membership("", [])
        self.spectral.setCurrentWidget(self.canvas)
        self.panel.cluster_page.lens_button.show()
        if self._sizes_before_lens:
            self.splitter.setSizes(self._sizes_before_lens)
        self.render_fitting()
        self.canvas.main.setFocus()
        self.guidance_changed.emit(self.guidance())

    def _update_lens(self) -> None:
        """Keep the lens on a recorded cluster by stable peak IDs, plane and display levels.

        The lens follows the selected peak's cluster. A selected peak that the run
        did not fit (one just added) keeps the lens on its cluster; after a refit,
        the lens follows the same peak and says how the cluster was regrouped.
        """
        result = self.fitting.result
        reviews = self.reviews() if result is not None else []
        by_id = {identity: review for review in reviews for identity in review.ids}
        review = by_id.get(self.selected_id or "")
        if review is None:
            review = self._context_cluster(reviews, by_id)
        if result is None or review is None:
            self.close_lens()
            return
        if self._lens_run is not None and result.run_id != self._lens_run:
            self._show_membership(review, by_id, result.run_id)
        elif self._membership_ids and self._membership_ids.isdisjoint(review.ids):
            self._membership_ids = frozenset()
            self.lens.set_membership("", [])
        self._lens_ids, self._lens_run = review.ids, result.run_id
        samples = next(s for s in result.completed_fit.samples if s.cluster_id == review.cluster_id)
        canvas = self.canvas
        comparison = self.fitting.comparison
        self.lens.show_cluster(
            result,
            current=bool(self.fitting.current),
            label=comparison.label,
            withdrawn=comparison.reason,
            added=comparison.added,
            departed=comparison.departed,
            ids=review.ids,
            names=review.names,
            selected=self.selected_id,
            peaks=self.peaks,
            plane=canvas.plane,
            levels=(
                canvas.contour_threshold,
                canvas.contour_factor,
                canvas.contour_count,
                canvas.residual_threshold,
            ),
            grid_indices=samples.grid_indices,
        )

    def _context_cluster(self, reviews, by_id):
        """The lens cluster for a selection the run did not fit (a peak just added).

        The lens keeps its cluster while that peak is in view, so adding a peak
        keeps its context; otherwise it shows the cluster with the nearest member.
        """
        peak = next((p for p in self.peaks if p.id == self.selected_id), None)
        current = next((r for r in reviews if r.ids == self._lens_ids), None)
        if peak is None or not reviews:
            if current is None and self._lens_ids:
                # A deleted selection: keep most of the lens members together.
                counts = Counter(by_id[i].cluster_id for i in self._lens_ids if i in by_id)
                if counts:
                    chosen = counts.most_common(1)[0][0]
                    return next(r for r in reviews if r.cluster_id == chosen)
            return current
        x, y = self.canvas.display(peak)
        (x0, x1), (y0, y1) = self.lens.panels[0].getViewBox().viewRange()
        if (
            current is not None
            and min(x0, x1) <= x <= max(x0, x1)
            and min(y0, y1) <= y <= max(y0, y1)
        ):
            return current
        axes = (self.canvas.x_axis, self.canvas.y_axis)
        points = {
            p.id: tuple(
                float(axis.ppm2pts(value))
                for axis, value in zip(axes, self.canvas.display(p), strict=True)
            )
            for p in self.peaks
        }
        here = points[peak.id]
        nearest = min(
            (i for i in by_id if i in points),
            key=lambda i: (points[i][0] - here[0]) ** 2 + (points[i][1] - here[1]) ** 2,
            default=None,
        )
        return by_id[nearest] if nearest is not None else current

    def _show_membership(self, review, by_id, run_id: str) -> None:
        """After a refit, say how the followed peak's cluster changed, by stable IDs.

        A peak the previous run fitted compares its previous and new clusters:
        joined members, members now fitted elsewhere (a split) and members not in
        this run. A peak added since says where it is now fitted, with a link back
        to the cluster that was on show.
        """
        names = {peak.id: peak.name for peak in self.peaks}
        followed = self.selected_id
        before = self._lens_ids
        if followed is not None and followed not in before and followed in review.ids:
            others = [names.get(i, i[:6]) for i in review.ids if i != followed]
            text = f"Run {run_id[:8]} fits {names.get(followed, followed[:6])} " + (
                "with " + ", ".join(others) if others else "on its own"
            )
            links = [] if set(before) & set(review.ids) else [(names.get(before[0], ""), before[0])]
            self._membership_ids = frozenset(review.ids) | frozenset(before)
            self.lens.set_membership(text + (" · previously shown:" if links else ""), links)
            return
        joined = [i for i in review.ids if i not in before]
        left = [i for i in before if i not in review.ids]
        if not joined and not left:
            self._membership_ids = frozenset()
            self.lens.set_membership("", [])
            return
        previous = [names.get(i, i[:6]) for i in before]
        label = previous[0] + (f" +{len(previous) - 1}" if len(previous) > 1 else "")
        elsewhere = [i for i in left if i in by_id]
        gone = [i for i in left if i not in by_id]
        parts = [f"Regrouped by run {run_id[:8]} · was {label}"]
        if joined:
            parts.append("joined by " + ", ".join(names.get(i, i[:6]) for i in joined))
        if gone:
            parts.append("not in this run: " + ", ".join(names.get(i, i[:6]) for i in gone))
        if elsewhere:
            parts.append("now fitted in other clusters:")
        self._membership_ids = frozenset(before) | frozenset(review.ids)
        self.lens.set_membership(" · ".join(parts), [(names.get(i, i[:6]), i) for i in elsewhere])

    def display_dialog(self) -> None:  # noqa: PLR0915
        """Display choices do not change scientific inputs or edit history."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Spectrum display")
        form = QtWidgets.QFormLayout(dialog)
        # Both contour sets are always offered: the lens draws experiment, model and
        # residual at once, and the overview may switch modes after the dialog.
        threshold = QtWidgets.QDoubleSpinBox()
        threshold.setRange(0.01, 1e12)
        threshold.setDecimals(4)
        threshold.setValue(self.canvas.contour_threshold)
        threshold.setKeyboardTracking(False)
        residual = QtWidgets.QDoubleSpinBox()
        residual.setRange(0.1, 1000)
        residual.setDecimals(2)
        residual.setValue(self.canvas.residual_threshold)
        residual.setKeyboardTracking(False)
        unexplained = QtWidgets.QDoubleSpinBox()
        unexplained.setRange(1, 100)
        unexplained.setDecimals(1)
        unexplained.setValue(self.unexplained_threshold)
        threshold.setAccessibleName("First contour")
        residual.setAccessibleName("First residual contour")
        unexplained.setAccessibleName("Unexplained signal beyond")
        for field in (threshold, residual, unexplained):
            field.setLocale(QtCore.QLocale.c())
        factor = QtWidgets.QDoubleSpinBox()
        factor.setRange(1.1, 5)
        factor.setValue(self.canvas.contour_factor)
        count = QtWidgets.QSpinBox()
        count.setRange(1, 20)
        count.setValue(self.canvas.contour_count)
        labels = QtWidgets.QCheckBox("Show peak labels")
        labels.setChecked(self.canvas.labels_visible)
        heavy = QtWidgets.QCheckBox("Heavier contours")
        heavy.setChecked(self.canvas.heavy_strokes)
        fit = QtWidgets.QPushButton("Fit current traces")
        fit.clicked.connect(self.canvas.fit_current_traces)
        for title, field in (
            ("First contour / a.u.", threshold),
            ("Level factor", factor),
            ("Levels per sign", count),
            ("First residual contour / σ_stat", residual),
            ("Unexplained signal beyond ± / σ_stat", unexplained),
            ("", labels),
            ("", heavy),
            ("Signed trace scale", fit),
        ):
            form.addRow(title, field)
        chosen_rules = self._highlight_fields(form)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.canvas.contour_threshold = threshold.value()
            self.canvas.residual_threshold = residual.value()
            self.canvas.contour_factor = factor.value()
            self.canvas.contour_count = count.value()
            self.canvas.labels_visible = labels.isChecked()
            self.canvas.heavy_strokes = heavy.isChecked()
            self.unexplained_threshold = unexplained.value()
            self._show_unexplained()
            self.canvas._refresh_labels()
            self.set_mode(self.canvas.comparison_mode)
            if self.lens_open:
                self._update_lens()
            chosen = chosen_rules()
            if chosen != self.rules:
                self.set_rules(chosen)

    def _highlight_fields(self, form: QtWidgets.QFormLayout):
        """Highlight thresholds in the Display dialog; returns a reader for them."""
        rules = self.rules
        highlights = QtWidgets.QLabel("REVIEW HIGHLIGHTS")
        highlights.setObjectName("section")
        form.addRow(highlights)

        def spin(value: float, low: float, high: float, suffix: str) -> QtWidgets.QDoubleSpinBox:
            field = QtWidgets.QDoubleSpinBox()
            field.setLocale(QtCore.QLocale.c())
            field.setRange(low, high)
            field.setDecimals(2)
            field.setSingleStep(0.05)
            field.setValue(value)
            field.setSuffix(suffix)
            return field

        correlation = spin(rules.correlation, 0.05, 1.0, "")
        low = spin(rules.linewidth_low, 0.05, 1.0, " × median")
        high = spin(rules.linewidth_high, 1.0, 20.0, " × median")
        for title, field in (
            ("Amplitude correlation |ρ| ≥", correlation),
            ("Linewidth below", low),
            ("Linewidth above", high),
        ):
            form.addRow(title, field)
        note = QtWidgets.QLabel(
            "Highlights are descriptive display rules; they never change results, "
            "warnings or run state."
        )
        note.setObjectName("caption")
        note.setWordWrap(True)
        form.addRow(note)
        return lambda: HighlightRules(correlation.value(), low.value(), high.value())

    def stop(self) -> None:
        """Stop display computation without blocking window close."""
        self._fit_clock.stop()
        self.fitting.stop()
        self.canvas.stop()
        self.lens.stop()
        self._unexplained_cancel.set()
        self._unexplained_worker.close()


def _recovery_note(error: str) -> str:
    """Why the accepted run would not survive a crash, and what keeps it."""
    return (
        f"The recovery copy of this run could not be written ({error}). "
        "Save the analysis to keep it if PeakFit closes unexpectedly."
    )


def _change_entries(changes: RunChanges, decimals: dict[str, int]) -> list[tuple]:
    """Compact page entries answering: fit, warnings, clustering, largest differences."""
    (chi_before, chi_after), (warned_before, warned_after) = changes.median_chi2, changes.warned
    before, after = changes.clusters
    clustering = f"Clusters {before} → {after}"
    if changes.regrouped:
        clustering += f" · {changes.regrouped} regrouped"
    if changes.added or changes.removed:
        clustering += f" · peaks +{changes.added} −{changes.removed}"
    # One short line per question: fit, warnings, clustering.
    entries: list[tuple] = [
        ("line", f"Median χ²ᵣ {chi_before:.2f} → {chi_after:.2f}"),
        ("line", f"Clusters with ▲ warnings {warned_before} → {warned_after}"),
        ("line", clustering),
    ]
    limit = 6
    if changes.changes:
        entries.append(("heading", "Clusters"))
    for change in changes.changes[:limit]:
        names = change.names
        label = names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "")
        if change.regrouped_from:
            lines = [f"Regrouped from {', '.join(change.regrouped_from)}"]
            if change.now:
                lines.append("Now " + ", ".join(change.now))
            if change.before:
                lines.append("Before regrouping " + ", ".join(change.before))
        else:
            lines = [f"{m} now reported" for m in change.now if m not in change.before]
            lines += [f"{m} no longer reported" for m in change.before if m not in change.now]
        entries.append(("link", label, "\n".join(lines), change.ids[0], True))
    if len(changes.changes) > limit:
        entries.append(("note", f"and {len(changes.changes) - limit} more"))
    if changes.displacements:
        entries.append(("heading", "Largest position differences"))
        for move in changes.displacements:
            delta = formatting.number(move.delta, decimals.get(move.axis, 4) + 1)
            entries.append(
                (
                    "link",
                    move.name,
                    f"δ {move.axis} {delta} ppm · {move.linewidths:.2f} LW",
                    move.working_id,
                    False,
                )
            )
        entries.append(
            ("note", "Differences between fitted positions; not a measure of fit quality.")
        )
    if changes.uncompared:
        entries.append(
            ("note", f"{changes.uncompared} peaks not compared: unusable in either run.")
        )
    return entries
