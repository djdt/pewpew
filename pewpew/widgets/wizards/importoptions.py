from PySide2 import QtCore, QtWidgets

import os.path

from pew import io

from typing import List, Tuple


class _ImportOptions(QtWidgets.QGroupBox):
    completeChanged = QtCore.Signal()

    def __init__(
        self, filetype: str, exts: List[str], parent: QtWidgets.QWidget = None
    ):
        super().__init__("Import Options", parent)
        self.filetype = filetype
        self.exts = exts

    def isComplete(self) -> bool:
        return True

    def updateOptions(self, path: str) -> None:
        pass


class AgilentOptions(_ImportOptions):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Agilent Batch", ["*.b"], parent)

        self.current_path = ""
        self.actual_datafiles = 0
        self.expected_datafiles = -1

        self.combo_dfile_method = QtWidgets.QComboBox()
        self.combo_dfile_method.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContents
        )
        self.combo_dfile_method.activated.connect(self.countDatafiles)
        self.combo_dfile_method.activated.connect(self.completeChanged)

        self.lineedit_dfile = QtWidgets.QLineEdit()
        self.lineedit_dfile.setReadOnly(True)

        self.check_name_acq_xml = QtWidgets.QCheckBox(
            "Read names from Acquistion Method."
        )

        dfile_layout = QtWidgets.QFormLayout()
        dfile_layout.addRow("Data File Collection:", self.combo_dfile_method)
        dfile_layout.addRow("Data Files Found:", self.lineedit_dfile)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(dfile_layout, 1)
        layout.addWidget(self.check_name_acq_xml, 0)
        self.setLayout(layout)

    def countDatafiles(self) -> None:
        method = self.combo_dfile_method.currentText()

        if method == "Alphabetical Order":
            datafiles = io.agilent.find_datafiles_alphabetical(self.current_path)
        elif method == "Acquistion Method":
            datafiles = io.agilent.acq_method_xml_read_datafiles(
                self.current_path,
                os.path.join(self.current_path, io.agilent.acq_method_xml_path),
            )
        elif method == "Batch Log CSV":
            datafiles = io.agilent.batch_csv_read_datafiles(
                self.current_path,
                os.path.join(self.current_path, io.agilent.batch_csv_path),
            )
        elif method == "Batch Log XML":
            datafiles = io.agilent.batch_xml_read_datafiles(
                self.current_path,
                os.path.join(self.current_path, io.agilent.batch_xml_path),
            )
        else:
            raise ValueError("Unknown data file collection method.")

        csvs = [
            os.path.join(d, os.path.splitext(os.path.basename(d))[0] + ".csv")
            for d in datafiles
        ]
        self.actual_datafiles = sum([os.path.exists(csv) for csv in csvs])
        self.expected_datafiles = len(datafiles)

        if self.expected_datafiles == 0:
            self.lineedit_dfile.clear()
        else:
            self.lineedit_dfile.setText(
                f"{self.actual_datafiles} ({self.expected_datafiles} expected)"
            )

    def isComplete(self) -> bool:
        return self.actual_datafiles > 0

    def setEnabled(self, enabled: bool) -> None:
        self.combo_dfile_method.setEnabled(enabled)
        self.check_name_acq_xml.setEnabled(enabled)

    def updateOptions(self, path: str) -> None:
        self.current_path = path
        current_text = self.combo_dfile_method.currentText()

        self.combo_dfile_method.clear()

        self.combo_dfile_method.addItem("Alphabetical Order")
        if os.path.exists(os.path.join(path, io.agilent.acq_method_xml_path)):
            self.combo_dfile_method.addItem("Acquistion Method")
            self.check_name_acq_xml.setEnabled(True)
        else:
            self.check_name_acq_xml.setEnabled(False)
        if os.path.exists(os.path.join(path, io.agilent.batch_csv_path)):
            self.combo_dfile_method.addItem("Batch Log CSV")
        if os.path.exists(os.path.join(path, io.agilent.batch_xml_path)):
            self.combo_dfile_method.addItem("Batch Log XML")

        # Restore the last method if available
        if current_text != "":
            self.combo_dfile_method.setCurrentText(current_text)
        else:
            self.combo_dfile_method.setCurrentIndex(self.combo_dfile_method.count() - 1)

        self.countDatafiles()


class NumpyOptions(_ImportOptions):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Numpy Archive", ["*.npz"], parent)
        self.check_calibration = QtWidgets.QCheckBox("Import calibration.")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.check_calibration)
        self.setLayout(layout)


class TextOptions(_ImportOptions):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Text Image", ["*.csv", "*.text", "*.txt"], parent)

        self.lineedit_name = QtWidgets.QLineEdit("_Isotope_")
        layout = QtWidgets.QFormLayout()
        layout.addRow("Isotope Name:", self.lineedit_name)

        self.setLayout(layout)

    def isComplete(self) -> bool:
        return self.lineedit_name.text() != ""


class ThermoOptions(_ImportOptions):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Thermo iCap Data", ["*.csv"], parent)

        self.radio_columns = QtWidgets.QRadioButton("Samples in columns.")
        self.radio_rows = QtWidgets.QRadioButton("Samples in rows.")

        self.combo_delimiter = QtWidgets.QComboBox()
        self.combo_delimiter.addItems([",", ";"])
        self.combo_decimal = QtWidgets.QComboBox()
        self.combo_decimal.addItems([".", ","])

        self.check_use_analog = QtWidgets.QCheckBox(
            "Use exported analog readings instead of counts."
        )

        layout_radio = QtWidgets.QVBoxLayout()
        layout_radio.addWidget(self.radio_columns)
        layout_radio.addWidget(self.radio_rows)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Export format:", layout_radio)
        layout.addRow("Delimiter:", self.combo_delimiter)
        layout.addRow("Decimal:", self.combo_decimal)
        layout.addRow(self.check_use_analog)
        self.setLayout(layout)

    def preprocessFile(self, path: str) -> Tuple[str, str, bool]:
        method = "unknown"
        has_analog = False
        with open(path, "r", encoding="utf-8-sig") as fp:
            lines = [next(fp) for i in range(3)]
            delimiter = lines[0][0]
            if "MainRuns" in lines[0]:
                method = "rows"
            elif "MainRuns" in lines[2]:
                method = "columns"
            for line in fp:
                if "Analog" in line:
                    has_analog = True
                    break
            return delimiter, method, has_analog

    def setEnabled(self, enabled: bool) -> None:
        self.combo_delimiter.setEnabled(enabled)
        self.combo_decimal.setEnabled(enabled)
        self.radio_rows.setEnabled(enabled)
        self.radio_columns.setEnabled(enabled)

    def updateOptions(self, path: str) -> None:
        delimiter, method, has_analog = self.preprocessFile(path)
        self.combo_delimiter.setCurrentText(delimiter)
        if method == "rows":
            self.radio_rows.setChecked(True)
        elif method == "columns":
            self.radio_columns.setChecked(True)

        if has_analog:
            self.check_use_analog.setEnabled(True)
        else:
            self.check_use_analog.setEnabled(False)
            self.check_use_analog.setChecked(False)
