from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QComboBox
from UIElements import Cell, Block
from enum import Enum, auto

class Cmds(Enum):
    NUM = auto()
    DEL = auto()
    MOUSE = auto()
    CELLCLICK = auto()
    IMPORT = auto()
    RESTART = auto()
    SOLVE = auto()
    FILLSINGLE = auto()
    UPDATE = auto()
    HIDSINGLE = auto()
    NAKEDPAIR = auto()
    POINTPAIR = auto()
    BOXLINE = auto()
    BOXTRIPLE = auto()
    XWING = auto()
    REGEN = auto()
    CLEAR = auto()

def GenButtonMap():
    return {
        Cmds.IMPORT: 'Import',
        Cmds.RESTART: 'Restart',
        Cmds.SOLVE: 'Solve',
        Cmds.FILLSINGLE: 'Fill Singles',
        Cmds.UPDATE: 'Update',
        Cmds.CLEAR: 'Clear'
    }

class PyQtSudokuView(QMainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()

        button_map = GenButtonMap()

        # Maps for commands and functions
        self.key_table = {k: Cmds.NUM for k in range(QtCore.Qt.Key_0, QtCore.Qt.Key_9 + 1)}
        self.key_table.update({QtCore.Qt.Key_Backspace: Cmds.DEL})
        self.func_map = {}

        board_layout, side_ui_layout = self.SetupWindow()

        # Variables (UI)
        self.cells = self.CreateBoard(self, board_layout)

        # Create message display
        self.msgText = QLabel('Solutions: ?')
        self.msgText.setStyleSheet("border: 1px solid black; font-size: 10px;")
        side_ui_layout.addWidget(self.msgText)

        # Create UI button elements
        for cmd in button_map:
            title = button_map[cmd]
            self.AddButton(side_ui_layout, title, lambda state, x=cmd: self.ExecuteCmd(x))

        # Add dropdown for secondary actions
        self.AddDropdown(side_ui_layout)

        # Add number pad
        self.AddNumberPad(side_ui_layout)

    def SetupWindow(self):
        """ Sets up the main window layout. """
        self.setGeometry(100, 100, 320, 480)  # Adjusted for 320x480 screen
        self.setWindowTitle("Simple Sudoku")
        self.setStyleSheet("background-color: grey;")

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        outer_layout = QGridLayout()
        board_layout = QGridLayout()
        side_ui_layout = QVBoxLayout()
        central_widget.setLayout(outer_layout)

        # Configure layout for a compact view
        outer_layout.addLayout(board_layout, 0, 0, 8, 8)
        outer_layout.addLayout(side_ui_layout, 0, 8, 4, 2)

        return board_layout, side_ui_layout

    @staticmethod
    def AddButton(layout, title, func):
        button = QPushButton(title)
        button.setStyleSheet("font-size: 10px; padding: 2px; margin: 1px;")
        button.clicked.connect(func)
        layout.addWidget(button)
        return button

    def AddDropdown(self, layout):
        """ Adds a dropdown for less-used functions """
        dropdown = QComboBox()
        dropdown.setStyleSheet("font-size: 10px; padding: 2px;")
        dropdown.addItem("More Actions...")
        dropdown.addItem("Highlight Hidden Singles")
        dropdown.addItem("Highlight Naked Pairs")
        dropdown.addItem("Highlight Pointing Pairs")
        dropdown.addItem("Highlight Box-Line Pairs")
        dropdown.addItem("Highlight Box Triples")
        dropdown.addItem("Highlight X-Wings")
        dropdown.currentIndexChanged.connect(self.DropdownSelected)
        layout.addWidget(dropdown)

    def DropdownSelected(self, index):
        """ Executes a command based on dropdown selection """
        dropdown_cmds = [
            None, Cmds.HIDSINGLE, Cmds.NAKEDPAIR, Cmds.POINTPAIR,
            Cmds.BOXLINE, Cmds.BOXTRIPLE, Cmds.XWING
        ]
        if index > 0:
            self.ExecuteCmd(dropdown_cmds[index])

    def AddNumberPad(self, layout):
        """ Adds a number pad for touchscreen input """
        num_pad_layout = QGridLayout()
        num_pad_buttons = []

        for i in range(1, 10):
            button = QPushButton(str(i))
            button.setStyleSheet("font-size: 10px; padding: 2px; margin: 1px;")
            button.clicked.connect(lambda state, x=i: self.NumPadClick(x))
            num_pad_buttons.append(button)

        for i in range(9):
            num_pad_layout.addWidget(num_pad_buttons[i], i // 3, i % 3)

        delete_button = QPushButton("Del")
        delete_button.setStyleSheet("font-size: 10px; padding: 2px; margin: 1px;")
        delete_button.clicked.connect(lambda: self.ExecuteCmd(Cmds.DEL))
        num_pad_layout.addWidget(delete_button, 3, 1)

        layout.addLayout(num_pad_layout)

    def NumPadClick(self, num):
        """ Handles number pad clicks """
        self.ExecuteCmd(Cmds.NUM, num)

    @staticmethod
    def CreateBlock(parent, layout, bi, bj):
        block = Block(parent)
        layout.addWidget(block, bi, bj)
        return block

    @staticmethod
    def CreateCell(i, j, boxes, click_func):
        bi, bj = i // 3, j // 3
        parent_box = boxes[bi][bj]
        cell = Cell(parent_box, i, j)
        cell.ConnectCelltoWindow(click_func)
        parent_box.AddCell(cell, i - 3 * bi, j - 3 * bj)
        return cell

    def CreateBoard(self, parent, layout):
        """ Creates board display with initial board values and candidates """
        blocks = [[self.CreateBlock(parent, layout, bi, bj) for bj in range(3)] for bi in range(3)]
        return [[self.CreateCell(i, j, blocks, self.CellClicked) for j in range(9)] for i in range(9)]

    def CellClicked(self, cell):
        """ Handles cell clicks """
        print(f"Cell at position ({cell.row}, {cell.column}) clicked.")
        self.ExecuteCmd(Cmds.CELLCLICK, cell)

    def ExecuteCmd(self, cmd, data=None):
        """ Executes a command based on `func_map` """
        if cmd in self.func_map:
            if data is not None:
                self.func_map[cmd](data)
            else:
                self.func_map[cmd]()

    def Connect(self, func_map):
        """ Connects functions to commands """
        self.func_map = func_map

    def keyPressEvent(self, event):
        """ Handles key presses """
        key = event.key()
        if key in self.key_table:
            cmd = self.key_table[key]
            if QtCore.Qt.Key_0 <= key <= QtCore.Qt.Key_9:
                num = int(key) - int(QtCore.Qt.Key_0)
                self.ExecuteCmd(cmd, num)
            else:
                self.ExecuteCmd(cmd)

    def mouseReleaseEvent(self, QMouseEvent):
        """ Handles mouse release events """
        print(f'({QMouseEvent.x()}, {QMouseEvent.y()}) - ({self.width()}, {self.height()})')
        self.ExecuteCmd(Cmds.MOUSE)

    def ResetAllCellsValid(self):
        """ Resets the validation of all cells """
        for i in range(9):
            for j in range(9):
                self.cells[i][j].SetValidity(is_invalid=False)

    def ShowInvalidCells(self, duplicate_cells):
        """ Highlights invalid cells """
        self.ResetAllCellsValid()
        for cell in duplicate_cells:
            self.cells[cell[0]][cell[1]].SetValidity(is_invalid=True)

    def UpdateAllCells(self, board, initial=False):
        """ Updates the values of all cells """
        for i in range(9):
            for j in range(9):
                self.cells[i][j].UpdateValue(board[i][j], initial)

    def UpdateAllCandidates(self, cand_board):
        """ Updates the candidates in unfilled cells """
        for i in range(9):
            for j in range(9):
                self.cells[i][j].UpdateCandidates(cand_board[i][j])

    def UpdateChangedCells(self, changed_cell_data):
        """ Updates only changed cells """
        for cell_info in changed_cell_data:
            i, j, n = cell_info
            self.cells[i][j].UpdateValue(n)

    def ClearHighlights(self):
        """ Clears all highlights """
        for i in range(9):
            for j in range(9):
                self.cells[i][j].ClearHilites()

    def HighlightRemovals(self, highlight_list):
        """ Highlights candidates for removal """
        for highlight_info in highlight_list:
            i, j = highlight_info.i, highlight_info.j
            self.cells[i][j].HiliteCandidates(highlight_info.candidates, colour='red')

    def HighlightValues(self, highlight_list):
        """ Highlights specific candidate values """
        for highlight_info in highlight_list:
            i, j = highlight_info.i, highlight_info.j
            self.cells[i][j].HiliteCandidates(highlight_info.candidates)

    def SetNumSolutions(self, num_solns=None):
        """ Sets the message area text """
        if num_solns == 1 or num_solns is None:
            self.msgText.setStyleSheet("border: 1px solid black; color: black;")
        else:
            self.msgText.setStyleSheet("border: 1px solid black; color: red;")

        self.msgText.setText('Solutions: ' + ('?' if num_solns is None else 'Invalid' if num_solns < 0 else str(num_solns)))
