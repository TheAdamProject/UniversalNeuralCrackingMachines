import os, glob, sys, importlib, pickle, random
from functools import partial
import tqdm

import myOS
from trainer import Trainer
from input_pipeline import _f_filter

def save_data(X, output_home, name_leak, name_run, data):
    outdir = os.path.join(output_home, name_leak)
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        out = os.path.join(outdir, 'X')
        with open(out, 'wb') as f:
            pickle.dump(X, f)
        
    out = os.path.join(outdir, name_run)
    with open(out, 'wb') as f:
        pickle.dump(data, f)
        

if __name__ == '__main__':
    try:
        conf_path_tester = sys.argv[1]
        conf_path = sys.argv[2]
        K = int(sys.argv[3])
    except:
        print("[USAGE] tester_setup model_setup num_tests")
        sys.exit(1)

    # load model
    conf = importlib.import_module(conf_path)
    hparams = conf.hparams
    name_run = conf_path.split('.')[1]
    print(name_run)

    trainer = Trainer(
        name_run,
        hparams['model_class'].make_models, 
        None,
        None,
        hparams
    )

    encoder, decoder = trainer.models
    input_fn = trainer.get_input_tensors
    
    # load tester
    tconf = importlib.import_module(conf_path_tester)
    hparams['testing'] = tconf.thparams
    tester = tconf.tester_class(encoder, decoder, input_fn, hparams) 
    output_home = tconf.output_home
    print(hparams['testing'])
    
    
    myOS.mkdir(output_home)
        
    # pick datasets
    random.seed(tconf.prg_seed)
    
    #paths = glob.glob(hparams['val_ds_dir'])
    paths = glob.glob(tconf.test_data)
    
    if tconf.filters:
        f_filter = partial(_f_filter, TD1=tconf.filters[0], TT=tconf.filters[1])
        paths = list(filter(f_filter, paths))

    random.shuffle(paths)
    paths = paths[:K]
    print(paths)
    
    # run tests
    for path in tqdm.tqdm(paths):
        name_leak = path.split('/')[-1]
        
        
        outdir = os.path.join(output_home, name_leak)
        _name_run = name_run + hparams['testing']['out_mod']


        out = os.path.join(outdir, _name_run)
        if os.path.isfile(out):
            print(f"Skipping: {out}")
            continue

        X, G, P, seed, pub_encoded = tester.compute_guess_numbers_from_file(path)

        out_data = {
            'G' : G,
            'P' : P,
            'seed' : seed,
            'n' : len(X),
            'sample_size' : hparams.get('sample_size'),
            'conf_path_tester' : conf_path_tester,
        }
        
        _name_run = name_run + hparams['testing']['out_mod']
        save_data(X, output_home, name_leak, _name_run, out_data)


