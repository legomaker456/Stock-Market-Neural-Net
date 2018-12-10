#Copied from Data science from Scratch by Joel Grus
class Table:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def __repr__(self):
        return str(self.columns)
