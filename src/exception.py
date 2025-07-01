"""
Custom exception handling for the stock portfolio project.
Provides a CustomException class for consistent error reporting.
"""
import sys

def error_message_detail(error, error_detail: sys):
    """
    Generate a detailed error message including file name and line number.
    Args:
        error: The exception object.
        error_detail (sys): The sys module for traceback info.
    Returns:
        str: Formatted error message with file and line info.
    """
    try:
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            error_message = "Error occurred in script [{0}] at line number [{1}] with message [{2}]".format(
                file_name, line_number, str(error)
            )
        else:
            # Handle case where there's no traceback (e.g., when raising a string)
            error_message = f"Error: {str(error)}"
        return error_message
    except:
        # Fallback if error_message_detail fails
        return f"Error: {str(error)}"

class CustomException(Exception):
    """
    Custom exception class for the stock portfolio project.
    Formats error messages with file and line information.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

