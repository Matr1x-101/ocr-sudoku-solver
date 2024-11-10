from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QGridLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal, pyqtSlot

Signal, Slot = pyqtSignal, pyqtSlot

def Value2String(value):
    """ Converts value to string for display. Shows empty space if value is not between 1 and 9 """
    return str(value) if (0 < value < 10) else ' '

class Candidate(QLabel):
    """ A label representing a candidate number within a cell """
    def __init__(self, str_value, parent):
        super(QLabel, self).__init__(str_value, parent)
        self.setStyleSheet("""
            Candidate[hilite="red"] {background-color: red;}
            Candidate[hilite="green"] {background-color: lightgreen;}
            Candidate[hilite="off"] {background: transparent;}
        """)
        self.SetHilite('off')
        self.setFont(QFont("Arial", 6))  # Font size optimized for smaller screens
        self.setAlignment(QtCore.Qt.AlignCenter)

    def SetHilite(self, hilite_colour='off'):
        """ Sets or removes highlight for this candidate """
        self.setProperty('hilite', hilite_colour)
        self.style().unpolish(self)
        self.style().polish(self)

class Cell(QLabel):
    """ A label representing a cell in the Sudoku grid """
    selected = Signal(object)

    def __init__(self, parent, i, j, value=0):
        super().__init__(Value2String(value), parent)
        self.i = i
        self.j = j

        # Styles for various cell states
        self.setStyleSheet("""
            Cell[selected="true"] {background-color: lightblue;}
            Cell[selected="false"] {background-color: white;}
            Cell[edit="true"] {color: darkgrey;}
            Cell[edit="false"] {color: black;}
            Cell[invalid="true"] {color: red;}
        """)
        self.setProperty('selected', False)
        self.SetEditStatus(value == 0)

        # Alignment and font adjustments
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFont(QFont("Arial", 18, QFont.Bold))  # Adjust font size for small screens

        # Create a grid layout for displaying candidates
        self.gridLayoutBox = QGridLayout()
        self.gridLayoutBox.setContentsMargins(0, 0, 0, 0)
        self.gridLayoutBox.setSpacing(1)
        self.setLayout(self.gridLayoutBox)

    @staticmethod
    def CandCoordFromValue(value):
        """ Returns grid coordinates for a candidate value (1-9) """
        return (value - 1) // 3, (value - 1) % 3

    def CreateCandidates(self, cand_set=None):
        """ Create grid of QLabel widgets to display the candidates """
        if cand_set is None:
            cand_set = set()

        for cand_value in range(1, 10):
            i, j = self.CandCoordFromValue(cand_value)
            cand_str = str(cand_value) if cand_value in cand_set else ' '
            cand_label = Candidate(cand_str, self)
            self.gridLayoutBox.addWidget(cand_label, i, j)

    def ConnectCelltoWindow(self, ClickFunc):
        """ Connect cell click event to a window function """
        self.selected.connect(ClickFunc)

    def CanEdit(self):
        return self.property('edit')

    def SetValidity(self, is_invalid):
        """ Sets the validity state of the cell """
        self.setProperty('invalid', is_invalid)
        self.style().unpolish(self)
        self.style().polish(self)

    def UpdateValue(self, value, initial=False):
        """ Updates the cell's value """
        str_value = Value2String(value)
        if initial:
            self.SetEditStatus(value == 0)

        if self.CanEdit() or initial:
            if value != 0:
                self.DeleteAllCandidates()
            elif self.gridLayoutBox.count() == 0:
                self.CreateCandidates()

            self.setText(str_value)

    def SetEditStatus(self, status):
        """ Sets whether the cell can be edited """
        self.setProperty('edit', status)
        self.style().unpolish(self)
        self.style().polish(self)

    def DeleteAllCandidates(self):
        """ Deletes all candidate widgets """
        for i in reversed(range(self.gridLayoutBox.count())):
            widget = self.gridLayoutBox.itemAt(i).widget()
            widget.setParent(None)
            widget.deleteLater()

    def UpdateCandidates(self, cand_set):
        """ Updates the candidates displayed in the cell """
        if self.text() == ' ':
            for cand_value in range(1, 10):
                i, j = self.CandCoordFromValue(cand_value)
                cand_widget = self.gridLayoutBox.itemAtPosition(i, j).widget()
                cand_str = str(cand_value) if cand_value in cand_set else ' '
                cand_widget.setText(cand_str)

    def HiliteCandidates(self, cand_set, colour='green'):
        """ Highlight specific candidates in the cell """
        if self.text() == ' ':
            for cand_value in iter(cand_set):
                i, j = self.CandCoordFromValue(cand_value)
                cand_widget = self.gridLayoutBox.itemAtPosition(i, j).widget()
                if cand_widget:
                    cand_widget.SetHilite(colour)

    def ClearHilites(self):
        """ Clear all candidate highlights in the cell """
        if self.text() == ' ':
            for cand_value in range(1, 10):
                i, j = self.CandCoordFromValue(cand_value)
                cand_widget = self.gridLayoutBox.itemAtPosition(i, j).widget()
                if cand_widget:
                    cand_widget.SetHilite('off')

    def mouseReleaseEvent(self, QMouseEvent):
        """ Handles cell being clicked """
        if not self.property('selected'):
            self.setProperty('selected', True)
            self.style().unpolish(self)
            self.style().polish(self)
        self.selected.emit(self)

    def Deselect(self):
        """ Deselects the cell """
        self.setProperty('selected', False)
        self.style().unpolish(self)
        self.style().polish(self)
    def FindCandidateClicked(self):
        """ Checks if a click occurred on any candidate and returns its value """
        if self.text() == ' ':
            for cand_value in range(1, 10):
                i, j = self.CandCoordFromValue(cand_value)
                cand_widget = self.gridLayoutBox.itemAtPosition(i, j).widget()
                if cand_widget and cand_widget.underMouse():
                    return cand_value
        return 0
    def AddCandidate(self, value):
        """ Adds candidate value from empty/unknown cell """
        if self.text() == ' ':
            i, j = self.CandCoordFromValue(value)
            cand_widget = self.gridLayoutBox.itemAtPosition(i, j).widget()
            cand_widget.setText(Value2String(value))
    
    def RemoveCandidate(self, value):
        """ Removes candidate value from empty/unknown cell """
        if self.text() == ' ':
            i, j = self.CandCoordFromValue(value)
            cand_widget = self.gridLayoutBox.itemAtPosition(i, j).widget()
            cand_widget.setText(' ')


class Block(QLabel):
    """ A container for cells, representing a Sudoku block """
    def __init__(self, parent):
        super(QLabel, self).__init__(parent)
        self.setStyleSheet('background-color: lightgrey;')

        self.gridLayoutBox = QGridLayout()
        self.gridLayoutBox.setContentsMargins(0, 0, 0, 0)
        self.gridLayoutBox.setSpacing(1)
        self.setLayout(self.gridLayoutBox)
        self.setFixedSize(100, 100)

    def AddCell(self, cell_QLabel, i, j):
        """ Adds a cell to the block """
        self.gridLayoutBox.addWidget(cell_QLabel, i, j)
