"""Track comparison results and controls, kept separate from removal selections."""

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit,
    QProgressBar, QPushButton, QTableView, QVBoxLayout, QWidget,
)

from .domain import normalize


class ComparisonModel(QAbstractTableModel):
    HEADERS = ("Ähnlichkeit", "Titel A", "Interpret A", "Datei A", "Titel B", "Interpret B", "Datei B")

    def __init__(self):
        super().__init__()
        self.matches = ()
        self.rows = []
        self.search = ""
        self.sort_column = 0
        self.sort_order = Qt.SortOrder.DescendingOrder

    def refresh(self):
        self.beginResetModel()
        needle = normalize(self.search)
        self.rows = [match for match in self.matches if not needle or needle in normalize(
            " ".join((match.first.title, match.first.artist, match.first.path,
                      match.second.title, match.second.artist, match.second.path)))]
        self.rows.sort(key=lambda match: self.values(match)[self.sort_column],
                       reverse=self.sort_order == Qt.SortOrder.DescendingOrder)
        self.endResetModel()

    @staticmethod
    def values(match):
        return (match.similarity, match.first.title, match.first.artist, match.first.path,
                match.second.title, match.second.artist, match.second.path)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.HEADERS[section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        match = self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{match.similarity:.2f} %" if index.column() == 0 else self.values(match)[index.column()]
        if role == Qt.ItemDataRole.ToolTipRole:
            return (f"{match.similarity:.4f} % Signalähnlichkeit im mittleren Fünf-Sekunden-Ausschnitt\n\n"
                    f"A · ID {match.first.id}\n{match.first.path}\n\n"
                    f"B · ID {match.second.id}\n{match.second.path}")

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        self.sort_column, self.sort_order = column, order
        self.refresh()


class ComparisonPage(QWidget):
    start_requested = Signal()
    cancel_requested = Signal()
    preview_requested = Signal(object)

    def __init__(self):
        super().__init__()
        self.result = None
        self.running = False
        self.model = ComparisonModel()
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
        self.preview_first = QPushButton("Track A anhören")
        self.preview_second = QPushButton("Track B anhören")
        self.preview_first.clicked.connect(lambda: self.preview_match(True))
        self.preview_second.clicked.connect(lambda: self.preview_match(False))
        toolbar.addWidget(self.preview_first)
        toolbar.addWidget(self.preview_second)
        layout.addLayout(toolbar)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for col in range(self.model.columnCount()):
            self.table.setColumnWidth(col, 105 if col == 0 else 190)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        self.table.selectionModel().currentRowChanged.connect(self.update_preview)
        self.model.modelReset.connect(self.update_preview)
        layout.addWidget(self.table, 1)
        self.notices = QPlainTextEdit()
        self.notices.setReadOnly(True)
        self.notices.setMaximumHeight(120)
        self.notices.setAccessibleName("Übersprungene Dateien beim Trackvergleich")
        layout.addWidget(self.notices)
        self.reset()

    def reset(self):
        self.result = None
        self.model.matches = ()
        self.model.refresh()
        self.status.setText("Noch kein Trackvergleich durchgeführt. Verglichen werden nur MP3-Dateien desselben Interpreten in der geöffneten Sammlung.")
        self.notices.clear()
        self.notices.hide()

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

    def finish(self, result):
        self.stop()
        self.result = result
        self.model.matches = result.matches
        self.model.refresh()
        count = len({track.id for match in result.matches for track in (match.first, match.second)})
        self.status.setText("Trackvergleich abgebrochen. Keine unvollständigen Ergebnisse übernommen." if result.cancelled else
                            f"{len(result.matches)} Trefferpaare mit mindestens 90 % · {count} Tracks · "
                            f"{result.checked} Titel geprüft · {len(result.skipped)} übersprungen")
        self.notices.setPlainText("Übersprungene Dateien:\n\n" + "\n\n".join(result.skipped))
        self.notices.setVisible(bool(result.skipped))

    def stop(self):
        self.running = False
        self.cancel_button.setEnabled(False)
        self.progress.hide()

    def set_available(self, has_library, busy):
        self.start_button.setEnabled(has_library and not busy)
        self.table.setEnabled(not busy)
        self.update_preview()

    def update_preview(self, *args):
        enabled = self.table.isEnabled() and self.table.currentIndex().isValid()
        self.preview_first.setEnabled(enabled)
        self.preview_second.setEnabled(enabled)

    def preview_match(self, first):
        index = self.table.currentIndex()
        if index.isValid() and self.table.isEnabled():
            match = self.model.rows[index.row()]
            self.preview_requested.emit(match.first if first else match.second)
