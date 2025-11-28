"""Lightweight event dispatcher for reporting long-running task progress."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable


class EventType(Enum):
    """Supported event types emitted by PeakFit services."""

    FIT_STARTED = auto()
    FIT_PROGRESS = auto()
    FIT_COMPLETED = auto()
    CLUSTER_STARTED = auto()
    CLUSTER_COMPLETED = auto()
    ERROR = auto()


@dataclass(slots=True)
class Event:
    """Base event carrying a type and arbitrary metadata."""

    event_type: EventType
    data: dict[str, Any]


@dataclass(slots=True)
class FitProgressEvent(Event):
    """Event emitted while iterating over clusters during fitting."""

    current_cluster: int
    total_clusters: int
    current_iteration: int
    total_iterations: int


class EventHandler(Protocol):
    """Protocol implemented by event handlers."""

    def handle(self, event: Event) -> None:  # pragma: no cover - thin interface
        """Process an incoming event."""


class EventDispatcher:
    """Simple pub-sub dispatcher for internal progress events."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler | Callable[[Event], None],
    ) -> None:
        """Register a handler for a particular event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        if callable(handler) and not hasattr(handler, "handle"):
            handler = _CallableHandler(handler)

        self._handlers[event_type].append(handler)

    def dispatch(self, event: Event) -> None:
        """Send an event to all subscribed handlers."""
        for handler in self._handlers.get(event.event_type, []):
            handler.handle(event)


class _CallableHandler:
    """Adapter that allows bare callables to act as event handlers."""

    def __init__(self, func: Callable[[Event], None]) -> None:
        self._func = func

    def handle(self, event: Event) -> None:  # pragma: no cover - trivial adapter
        self._func(event)
