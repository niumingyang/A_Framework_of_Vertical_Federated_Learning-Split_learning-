from configparser import ConfigParser
import logging
import importlib

from utils.logger import initLogger, print_config
from privacy import model_completion

# start only model completion attack
MC_ONLY = False

# parse config
cfg = ConfigParser()
cfg.read('vfl_config.cfg')
# print(cfg.sections())
# print(cfg.options('dataset'))
# print(cfg.items('dataset'))
# print(cfg['dataset'])
# print(cfg['dataset'])

dataset_name = cfg['dataset']['name']

# initialize logger
config = dataset_name
if int(cfg['privacy']['dcorLoss']) == 1:
    config = config + '_dcor=' + cfg['privacy']['dcorCoef']
if int(cfg['privacy']['labelConfusion']) == 1:
    config = config + '_cfs'
save_dir = './results/' + dataset_name + '/'
save_log_path = save_dir + config + '.log'
save_model_path = save_dir + config + '.pth'
log_level = logging.DEBUG

if MC_ONLY == False:
    initLogger(level=log_level, log_file=save_log_path)
    # print config
    print_config(cfg, 'vFL config:')

    #  get model
    logging.info('>>> Prepare for vFL training...')
    module = importlib.import_module('models.' + dataset_name)
    class_obj = getattr(module, dataset_name.capitalize() + 'Model')
    model = class_obj(cfg)

    # train and eval model
    logging.info('')
    logging.info('>>> vFL training and evaluating starts...')
    model.train_model()

    # save model
    logging.info('')
    logging.info('>>> Saving model...')
    model.save_model(save_model_path)
    logging.info('The model is saved in ' + save_model_path)

# model completion
if int(cfg['privacy']['modelCompletion']) == 1:
    save_log_path = save_dir + config + '_mc.log'
    initLogger(level=log_level, log_file=save_log_path)
    logging.info('')
    logging.info('>>> Model completion...')
    print_config(cfg, 'model completion config:')
    module = importlib.import_module('datasets.' + dataset_name + '.' + dataset_name + '_data')
    class_obj = getattr(module, dataset_name.capitalize() + 'ForSSL')
    dataset = class_obj(cfg)
    mc = model_completion.ModelCompletion(
        resume=save_model_path, dataset=dataset, config=cfg)
    mc.ssl_training()
