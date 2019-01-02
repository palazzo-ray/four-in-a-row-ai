import logging

logging.basicConfig(filename='training.log' ,
                    level=logging.INFO,
                    format= '%(asctime)s - %(levelname)s - %(module)s - %(message)s' , datefmt='%Y-%m-%d %H:%M:%S')
#logging.basicConfig(filename='training.log' ,format= '%(asctime)s - %(levelname)s - %(message)s' , datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('main')