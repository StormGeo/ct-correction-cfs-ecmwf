import os
import time
import logging
import logging.config
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + '/ct-near'

PROVIDER_LIST = [
                    'inmet',
                    'cemaden',
                    'vale',
                    'metos',
                    'ciram',
                    'simepar',
                    'simagro',
                    'enel',
                    'inema',
                    'cgesp',
                    'funceme',
                    'scicrop',
                    'ciiagro',
                    'omega',
                    'mrs',
                    'ecosoft',
                    'redemet',
                    'ogimet',
                    'arable',
                    'ideam'
                ]

PROVIDER_TO_VALID = ['inmet']
def load_config(vname):
    path = f'{BASE_DIR}/configs/{vname}.yml'
    with open(path, 'r') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise e
        return data

def config_log():
    """
    Setup the logging routine.

    This function configures the logging setup for the application. It defines a dictionary
    `LOGGING_CONFIG` containing the configuration parameters for the logging system,
    including formatters, handlers, and loggers. The configuration is based on the following:

    - Log messages are formatted with the 'standard' formatter, which includes the log level,
      timestamp, and the actual log message.
    - A console handler is set up to handle INFO-level messages and above, using the 'standard'
      formatter.
    - The root logger is configured to use the console handler, with a log level of DEBUG.

    Additionally, the function sets up the logging configuration using the `dictConfig` method
    from the `logging.config` module. It also adjusts the `Formatter.converter` to use
    Greenwich Mean Time (GMT).

    Usage:
    - Call this function once at startup to configure the logging system.

    Note:
    - The logging configuration is intended to be included in each module where logging is
      used by calling `logging.getLogger('')` to get the root logger.

    Example:
    ```python
    import logging
    import time

    # Call config_log() at the beginning of your script or application
    config_log()

    # Use logging in your modules
    logger = logging.getLogger(__name__)
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    ```

    """

    # loggin ...................................
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'standard': {
                'format': '[%(levelname)s @ %(asctime)s] %(message)s',
                'datefmt': '%Y-%m-%dT%H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler'
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ["console"],
                'level': 'DEBUG',
                'propagate': False
            },
        }
    }

    # Run once at startup:
    logging.config.dictConfig(LOGGING_CONFIG)

    # Include in each module:
    logging.getLogger('')
    logging.Formatter.converter = time.gmtime


def ensure_interpol(method_on_exception='nearest'):
    def decorator(func):
        def wrapper(*args, interpol_method=None, **kwargs):
            try:
                return func(*args, interpol_method=interpol_method, **kwargs)
            except Exception as e:
                time.sleep(1)
                logging.warning(f"src.commons.ensure_interpol >> an exception happened {e} @ "
                                f"using {method_on_exception} to ensure the result")
                return func(*args, interpol_method=method_on_exception, **kwargs)

        return wrapper

    return decorator