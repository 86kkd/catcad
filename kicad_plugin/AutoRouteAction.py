"""
KiCad ActionPlugin stub for RL Auto Router.
This is a Phase 0 placeholder to validate packaging and CI only.
"""

try:
    import pcbnew  # type: ignore
except Exception:  # pragma: no cover - not available in CI
    pcbnew = None  # type: ignore


def register_plugin() -> None:
    """Register the plugin with KiCad if available (no-op in CI)."""
    if pcbnew is None:
        return

    # Minimal ActionPlugin skeleton
    class AutoRouteAction(pcbnew.ActionPlugin):  # type: ignore[attr-defined]
        def defaults(self) -> None:
            self.name = "RL Auto Router"
            self.category = "Routing"
            self.description = "Auto-route selected nets using RL/heuristics"

        def Run(self) -> None:
            # Placeholder: In Phase 4, call ONNX inference and write tracks
            pass

    AutoRouteAction().Register()


if __name__ == "__main__":
    register_plugin()
