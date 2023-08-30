import configparser
import pathlib


from arpreprocessing.amigos import Amigos
from arpreprocessing.ascertain import Ascertain
from arpreprocessing.decaf import Decaf
from arpreprocessing.wesad import Wesad
from utils.loggerwrapper import GLOBAL_LOGGER

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")

    # config_path = pathlib.Path(__file__).parent.absolute() / "config.ini"
    # print(pathlib.Path(__file__).parent.absolute())
    # config = configparser.ConfigParser()
    # config.read(config_path)

    dataset = Wesad(GLOBAL_LOGGER, config['Paths']['wesad_dir']).get_dataset()
    dataset.save(config['Paths']['mts_out_dir'])

    # dataset = Decaf(GLOBAL_LOGGER, config['Paths']['decaf_dir']).get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    # dataset = Ascertain(GLOBAL_LOGGER, config['Paths']['ascertain_dir']).get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    # dataset = Amigos(GLOBAL_LOGGER, config['Paths']['amigos_dir']).get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])
