import logging
import sys

winlog = logging.getLogger("winrain")
winlog.setLevel(logging.DEBUG)

tl = logging.StreamHandler()

formatter = logging.Formatter(
        fmt="[%(asctime)s +%(msecs)d] [%(process)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %X"
        )

tl.setFormatter(formatter)

_log_id = "0"

winlog.addHandler(tl)

def info(log_str):
    try:
        raise Exception
    except:
        f = sys.exc_info()[2].tb_frame.f_back
    path_info = f.f_code.co_filename + ":" + str(f.f_lineno)
    winlog.info(path_info + " log_id:" + _log_id + " "+ log_str)

def warning(log_str):
    try:
        raise Exception
    except:
        f = sys.exc_info()[2].tb_frame.f_back
    path_info = f.f_code.co_filename + ":" + str(f.f_lineno)
    winlog.warning(path_info + " log_id:" + _log_id + " "+ log_str)

def debug(log_str):
    try:
        raise Exception
    except:
        f = sys.exc_info()[2].tb_frame.f_back
    path_info = f.f_code.co_filename + ":" + str(f.f_lineno)
    winlog.debug(path_info + " log_id:" + _log_id + " "+ log_str)

def error(log_str):
    try:
        raise Exception
    except:
        f = sys.exc_info()[2].tb_frame.f_back
    path_info = f.f_code.co_filename + ":" + str(f.f_lineno)
    winlog.error(path_info + " log_id:" + _log_id + " "+ log_str)

def _set_log_id(log_id):
    global _log_id
    _log_id = log_id
    

