class PewPewError(Exception):
    pass


class PewPewCalibrationError(PewPewError):
    pass


class PewPewConfigError(PewPewError):
    pass


class PewPewDataError(PewPewError):
    pass


class PewPewFileError(PewPewError):
    pass
