class PewPewError(Exception):
    pass


class PewPewFileError(PewPewError):
    pass


class PewPewDataError(PewPewError):
    pass


class PewPewConfigError(PewPewError):
    pass
