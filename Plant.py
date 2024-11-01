class Plant:
    def __init__(self, name, number, rows, facilities, matrix):
        self.name = name
        self.number = number
        self.rows = rows
        self.facilities = facilities
        self.matrix = matrix

    def __str__(self):
        return f"Plant {self.name} with {self.number} facilities and {self.rows} rows"

