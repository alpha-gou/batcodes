import logging


# add two logger handler, export to terminal and logfile
def get_logger(
    name: str = "logger",
    level=logging.DEBUG,
    logfile: str = None,
    logmode: str = "w",  # w 清空, a 追加
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # https://stackoverflow.com/questions/533048/how-to-log-source-file-name-and-line-number-in-python
    formatter = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file
    if logfile is not None:
        print(f"save log to:", logfile)
        file_handler = logging.FileHandler(logfile, mode=logmode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
