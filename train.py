import sys, importlib, pprint

from trainer import Trainer
from input_pipeline import make_dataset
from UNCM import export_models


if __name__ == '__main__':
    try:
        setup_module = sys.argv[1]
        name_run = setup_module.split('.')[-1]
    except:
        print("[USAGE] setup_module(e.g., exp_setups.UNCM)")
        sys.exit(1)

    setup = importlib.import_module(setup_module)
    hparams = setup.hparams    
    pprint.pprint(hparams)
    
    
    # load datasets
    ds_train, N = make_dataset(hparams['train_ds_dir'], hparams, hparams['conditional'], filters=hparams['filters_dataset'])
    ds_val, _ = make_dataset(hparams['val_ds_dir'], hparams, hparams['conditional'], filters=hparams['filters_dataset'])
    print(f'Number of leaks in train set: {N}')

    # setup model
    trainer = Trainer(
        name_run,
        hparams['model_class'].make_models, 
        ds_train,
        ds_val,
        hparams
    )

    # train
    trainer()
    
    export_models(hparams, name_run, trainer.models)