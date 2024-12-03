import sys
from src.logger import get_logger

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return f'Error has occured at [{file_name}] at [{line_number}] -> {str(error)}'

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):

        super().__init__(error_message)

        self.error_detail = error_detail
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
    
if __name__ == '__main__':

    logger = get_logger('khatarnak')
    try:
        a = float(input("Enter Here: "))

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)