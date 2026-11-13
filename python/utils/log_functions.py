import os, datetime, logging
from pathlib import Path
import sys

def setup_logger(name : str, log_dir : Path, subfolder_name : str, log_file : Path, level : int = logging.INFO, mode : str = "w") -> logging.Logger:
    """
    https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
    Args:
        name (str): name of the log
        log_dir (Path): path to the log file
        subfolder_name (str): name of the subfolder
        log_file (Path): name of the log file
        level (Optional[int], optional): Defaults to logging.INFO.
        mode (Optional[str], optional): Defaults to "w".
    Returns:
        logging.Logger: logger
    """
    os.makedirs(os.path.join(log_dir, subfolder_name), exist_ok=True)

    formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_dir, subfolder_name, log_file), mode = mode)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    now = datetime.datetime.now()
    logger.info(f"Logger initialized at {now.strftime('%Y/%m/%d %H:%M:%S')}")
    
    return logger
