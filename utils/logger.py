import logging

def initLogger(level, log_file):
    logging.basicConfig(level=level)
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def print_config(cfg, header):
    logging.info(header)
    for sec in cfg.sections():
        logging.info('[' + sec + ']')
        for (key, value) in cfg.items(sec):
            logging.info('    ' + key + ' = ' + value)
