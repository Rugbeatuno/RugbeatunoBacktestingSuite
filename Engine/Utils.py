def dollar_format(value: float) -> str:
    """
    Formats a float value as a dollar price string.

    Args:
        value (float): The value to format.

    Returns:
        str: The formatted dollar price string.
    """
    return f"${value:,.2f}"


def percentage_format(value: float) -> str:
    """
    Formats a float value as a percentage string.

    Args:
        value (float): The value to format.
    """
    return f'{(100 * value):.2f}%'
