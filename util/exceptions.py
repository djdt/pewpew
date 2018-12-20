class PewPewError(Exception):
    pass


class PewPewImportError(PewPewError):
    def __init__(self, message, file):
        self.file = file
        self.message = message
