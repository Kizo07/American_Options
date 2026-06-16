def validate_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_choice(name, value, choices):
    if value not in choices:
        allowed = ", ".join(repr(choice) for choice in sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}")


def validate_rho(rho):
    if not -1 <= rho <= 1:
        raise ValueError("rho must be between -1 and 1")
