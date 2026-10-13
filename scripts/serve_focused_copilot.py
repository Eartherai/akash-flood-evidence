"""Serve the focused development adapter using the verified case HTTP service.

The preserved first adapter/server and its measured run remain unchanged.
This launcher changes retrieval/prompting only, not the model or hazard outputs.
"""
from pathlib import Path

import serve_fresh_copilot as service
from akash_flood.focused_copilot import FocusedCaseFactStore, ask_focused_model
from build_fresh_report import record


class FocusedCaseService(service.CaseService):
    def __init__(self, **options):
        super().__init__(**options)
        self.bindings.append(record(Path(__file__)))


if __name__ == '__main__':
    service.FreshCaseFactStore = FocusedCaseFactStore
    service.ask_local_model = ask_focused_model
    service.CaseService = FocusedCaseService
    service.main()
