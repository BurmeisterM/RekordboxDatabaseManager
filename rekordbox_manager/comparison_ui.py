"""Controls and printable error report for the shared track results view."""

from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextDocument, QTextOption
from PySide6.QtPrintSupport import QPrinter, QPrintPreviewDialog
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)


class ComparisonPage(QWidget):
    start_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, model):
        super().__init__()
        self.result = None
        self.running = False
        self.model = model
        self.library_path = ""
        self.finished_at = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.start_button = QPushButton("Trackvergleich starten")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start_requested)
        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.request_cancel)
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.cancel_button)
        toolbar.addStretch()
        self.print_button = QPushButton("Fehlerhafte Dateien drucken…")
        self.print_button.setToolTip("Druckvorschau mit Titel, Interpret, ID, vollständigem Dateipfad und Fehlergrund öffnen.")
        self.print_button.clicked.connect(self.print_errors)
        toolbar.addWidget(self.print_button)
        layout.addLayout(toolbar)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        self.notices = QPlainTextEdit()
        self.notices.setReadOnly(True)
        self.notices.setMaximumHeight(100)
        self.notices.setAccessibleName("Übersprungene Dateien beim Trackvergleich")
        layout.addWidget(self.notices)
        self.reset()

    def reset(self):
        self.result = None
        self.library_path = ""
        self.finished_at = ""
        self.model.set_library(None)
        self.status.setText("Noch kein Trackvergleich durchgeführt. Verglichen werden nur MP3-Dateien desselben Interpreten in der geöffneten Sammlung.")
        self.notices.clear()
        self.notices.hide()
        self.print_button.setEnabled(False)

    def start(self):
        self.reset()
        self.running = True
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status.setText("Die mittleren Audioausschnitte werden gelesen …")

    def request_cancel(self):
        self.cancel_button.setEnabled(False)
        self.status.setText("Trackvergleich wird abgebrochen …")
        self.cancel_requested.emit()

    def report_progress(self, current, total, message):
        if not self.running:
            return
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(current)
        if self.cancel_button.isEnabled():
            self.status.setText(message)

    def finish(self, result, library):
        self.stop()
        self.result = result
        self.library_path = str(library.path)
        self.finished_at = datetime.now().astimezone().strftime("%d.%m.%Y %H:%M %Z")
        self.model.set_library(library)
        count = sum(len(group.tracks) for group in library.groups)
        self.status.setText("Trackvergleich abgebrochen. Keine unvollständigen Ergebnisse übernommen." if result.cancelled else
                            f"{len(library.groups)} Gruppen · {count} Tracks · {len(result.matches)} Trefferpaare mit mindestens 90 % · "
                            f"{result.checked} Titel geprüft · {len(result.skipped)} übersprungen · {len(result.errors)} Dateifehler")
        self.notices.setPlainText("Übersprungene Dateien:\n\n" + "\n\n".join(result.skipped))
        self.notices.setVisible(bool(result.skipped))
        self.print_button.setEnabled(bool(result.errors))

    def stop(self):
        self.running = False
        self.cancel_button.setEnabled(False)
        self.progress.hide()

    def set_available(self, has_library, busy):
        self.start_button.setEnabled(has_library and not busy)
        self.print_button.setEnabled(bool(self.result and self.result.errors) and not busy)

    def error_document(self):
        document = QTextDocument()
        document.setDefaultFont(QFont("Segoe UI", 10))
        option = document.defaultTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        document.setDefaultTextOption(option)
        errors = self.result.errors if self.result else ()
        heading = ("Rekordbox Database Manager – Fehlerhafte Dateien\n\n"
                   f"Sammlung: {self.library_path}\nTrackvergleich vom {self.finished_at}\n"
                   f"{len(errors)} fehlerhafte Dateieinträge\n")
        if self.result and self.result.cancelled:
            heading += "Vergleich abgebrochen – diese Liste enthält nur die bis dahin erkannten Dateifehler.\n"
        document.setPlainText(heading + "\n" + "\n\n".join(
            f"{index}. {error}" for index, error in enumerate(errors, 1)))
        return document

    def print_errors(self):
        if not self.result or not self.result.errors:
            return
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setDocName("Trackvergleich – Fehlerhafte Dateien")
        document = self.error_document()
        preview = QPrintPreviewDialog(printer, self)
        preview.setWindowTitle("Fehlerhafte Dateien – Druckvorschau")
        preview.resize(1000, 750)
        preview.paintRequested.connect(document.print_)
        preview.exec()
