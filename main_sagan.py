
from sagan.parameter import *
from sagan.trainer import Trainer
#from tester import Tester
from sagan.data_loader import Data_Loader
from torch.backends import cudnn
from sagan.utils import make_folder

def main(config):
    # For fast training
    cudnn.benchmark = True

    #data loader variables
    train=True
    dataset=None
    path=None
    imsize=None
    batch_size=None
    shuf=False
    model='sagan'


    # Data loader
    data_loader = Data_Loader(train, dataset, path, imsize,
                             batch_size, shuf=shuf)

    # Create directories if not exist
    # make_folder(config.model_save_path, config.version)
    # make_folder(config.sample_path, config.version)
    # make_folder(config.log_path, config.version)
    # make_folder(config.attn_path, config.version)


    if train:
        if model=='sagan':
            trainer = Trainer(data_loader.loader(), config)
        elif model == 'qgan':
            trainer = qgan_trainer(data_loader.loader(), config)
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), config)
        tester.test()

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)