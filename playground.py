import logging

FORMAT = "{%(asctime)s %(name)s} %(message)s"
logging.basicConfig(level=logging.INFO,format=FORMAT,datefmt="[%Y-%m-%d %H:%M:%S]")

class TestLogger:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def print_log(self):
        self.logger.info('testing logger')

if __name__ == '__main__':
    tl = TestLogger()
    tl.print_log()
    print('finish')