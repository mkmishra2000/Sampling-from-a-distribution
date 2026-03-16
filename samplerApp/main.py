"""
Entry point for the Sampler App.

Usage:
    python -m samplerApp.main
    # or from workspace root:
    python run_app.py
"""

import sys
from PySide6.QtWidgets import QApplication
from .main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
