import configparser
import pathlib


from arpreprocessing.amigos import Amigos
from arpreprocessing.ascertain import Ascertain
from arpreprocessing.decaf import Decaf
from arpreprocessing.wesad import Wesad
from arpreprocessing.kemowork import KEmoWork
from GeneralizedModel.utils.loggerwrapper import GLOBAL_LOGGER

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")

    # dataset = Decaf(GLOBAL_LOGGER, config['Paths']['decaf_dir']).get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    # dataset = Ascertain(GLOBAL_LOGGER, config['Paths']['ascertain_dir']).get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    # dataset = Amigos(GLOBAL_LOGGER, config['Paths']['amigos_dir']).get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    dataset = Wesad(GLOBAL_LOGGER, config['Paths']['wesad_dir']).get_dataset()
    dataset.save(config['Paths']['mts_out_dir'])

    # dataset = KEmoWork(GLOBAL_LOGGER, config['Paths']['kemowork_dir'], 'STRESS').get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])