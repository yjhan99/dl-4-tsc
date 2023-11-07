import configparser
import pathlib

from arpreprocessing.wesad import Wesad
from arpreprocessing.kemowork import KEmoWork
from arpreprocessing.case import Case
from arpreprocessing.kemocon import KEmoCon
from arpreprocessing.dreamer import Dreamer
from utils.loggerwrapper import GLOBAL_LOGGER

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")

    # dataset = Decaf(GLOBAL_LOGGER, config['Paths']['decaf_dir']).get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    # dataset = Ascertain(GLOBAL_LOGGER, config['Paths']['ascertain_dir']).get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    # dataset = Amigos(GLOBAL_LOGGER, config['Paths']['amigos_dir']).get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    # dataset = Wesad(GLOBAL_LOGGER, config['Paths']['wesad_dir']).get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    # dataset = KEmoWork(GLOBAL_LOGGER, config['Paths']['kemowork_dir'], 'STRESS').get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    # dataset = Case(GLOBAL_LOGGER, config['Paths']['case_dir'], 'AROUSAL').get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    # dataset = KEmoCon(GLOBAL_LOGGER, config['Paths']['kemocon_dir'], 'arousal').get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    dataset = Dreamer(GLOBAL_LOGGER, config['Paths']['dreamer_dir']).get_dataset()
    dataset.save(config['Paths']['mts_out_dir'])
