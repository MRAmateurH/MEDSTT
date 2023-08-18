import logging

# def init(log_file='test.log'):
#     '''
#     Initializes the Log object.

#     Parameters:
#         log_file (str): The name of the log file. Default is 'test.log'.
#     '''
#     global logger
    
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)
    
#     # Configure the logging settings
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format='%(asctime)s - %(module)-15s - %(levelname)s - %(message)s',
#         filename=log_file,
#         filemode='w',
#         encoding='utf-8'
#     )

def init(log_file='test.log'):
    '''
    Initializes the Log object.

    Parameters:
        log_file (str): The name of the log file. Default is 'test.log'.
    '''
    global logger

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a file handler to save logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a stream handler to display logs in the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Define the log format
    formatter = logging.Formatter('%(asctime)s - %(module)-15s - %(levelname)s - %(message)s')

    # Set the formatter for the handlers
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# class log():
    

#     def __init__(self) -> None:
#         pass

#     def init(self, log_file='test.log'):
#         '''
#         Initializes the Log object.

#         Parameters:
#             log_file (str): The name of the log file. Default is 'test.log'.
#         '''
#         global logger

#         logger = logging.getLogger()
#         logger.setLevel(logging.DEBUG)

#         # Create a file handler to save logs to a file
#         file_handler = logging.FileHandler(log_file)
#         file_handler.setLevel(logging.DEBUG)

#         # Create a stream handler to display logs in the console
#         stream_handler = logging.StreamHandler()
#         stream_handler.setLevel(logging.INFO)

#         # Define the log format
#         formatter = logging.Formatter('%(asctime)s - %(module)-15s - %(levelname)s - %(message)s')

#         # Set the formatter for the handlers
#         file_handler.setFormatter(formatter)
#         stream_handler.setFormatter(formatter)

#         # Add the handlers to the logger
#         logger.addHandler(file_handler)
#         logger.addHandler(stream_handler)