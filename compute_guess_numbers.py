import sys, importlib, math

import UNCM
from tester import Tester

def export_results(out_path, passwords, guess_numbers, probabilities):
    assert len(passwords) == len(guess_numbers) == len(probabilities)
    with open(out_path, 'w') as f:
        for password, guess_number, probability in zip(passwords, guess_numbers, probabilities):
            print(f"{password}\t{probability}\t{math.ceil(guess_number)}", file=f)
            

if __name__ == '__main__':
    try:
        conf_path = sys.argv[1]
        in_path = sys.argv[2]
        out_path = sys.argv[3]
    except:
        print("[USAGE] model_conf_file input_credetial_file output_file")
        sys.exit(1)

    # parse conf
    name_run = conf_path.split('.')[1]
    conf = importlib.import_module(conf_path)
    hparams = conf.hparams
    
    # load pre-trained UNCM
    print(f"Loading {name_run}...")
    uncm = UNCM.import_models(hparams, name_run)
    conf_encoder_uncm, cpassmodel_uncm = uncm
    input_fn = UNCM.make_get_input_tensors(hparams)
   
    # setup tester
    tester = Tester(conf_encoder_uncm, cpassmodel_uncm, input_fn, hparams)    
            
    passwords, guess_numbers, probabilities, seed = tester.compute_guess_numbers_from_file(in_path)
    
    print(f"Saving in {out_path}")
    export_results(out_path, passwords, guess_numbers, probabilities)

