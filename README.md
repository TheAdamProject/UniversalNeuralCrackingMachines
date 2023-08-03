



# 🌎 Universal Neural-Cracking-Machines  (UNCMs) 
Code and pre-trained models for the paper: *"Universal Neural-Cracking-Machines: Self-Configurable Password Models from Auxiliary Data"*  [🎉IEEE S&P'24🎉]  by [Dario Pasquini](https://pasquini-dario.github.io/me/), [Giuseppe Ateniese](https://ateniese.github.io), and [Carmela Troncoso](http://carmelatroncoso.com).

🔥 **All  the main functionalities are now available** 🔥, but we are planning to add more stuff:

#### Long term goals:

- [ ] Adding JavaScript interface to use seeded password models in the browser.
  - [ ] Adding compressed pre-trained models for browser inference

- [ ] Adding improved models trained on (hard) synthetic password leaks 
- [ ] Adding bigger pre-trained UNCMs for server-side estimation

## Download pre-trained models: 
Currently, we offer three models aimed at reproducing the results in the paper:

* [UNCM_medium_8096con_2048pm](https://drive.google.com/drive/folders/1Xf549jF6zo2zlZ4ZbfH3cxN_kEpSZm4K?usp=share_link): The standard UNCM
* [Baseline](https://drive.google.com/drive/folders/19u2Ld3PWIvRZ9ejYCVD9dcoGsl6e8pKN?usp=share_link): Baseline model (non-conditional password model)
* [DP-UNCM](https://drive.google.com/drive/folders/1wWi71UJrcObwoBt9GkH0Q-nbnDan_M4Y?usp=share_link): UNCM trained to handle DP configuration seeds

## How to use UNCMs:

We have 3 playground notebooks: 

* *sampling_example.ipynb*: Notebook on how to create and sample passwords from a seeded password models (not meant for password guessing).
* *get_probability.ipynb*: Notebook on how to assign probabilities to plaintext passwords with a seeded password model. 
* *get_guess_numbers.ipynb*: Notebook on how to assign guess numbers to plaintext passwords with a  seeded password model. 

### Compute guess numbers on plaintext passwords:

The script *'compute_guess_numbers.py'* utilizes the Monte Carlo estimation method by Dell’Amico et al. to calculate guess numbers (and probabilities too) for a set of plaintext credentials.

In order to use it, you first need to:

* Download a pertained model 

* Put it in the directory *'./keras_models'* keeping the original folder with the same name as in the configuration files in *'configs'* e.g.:

  ```
  keras_models
  ├── baseline
  │   └── password_model.h5
  ├── UNCM_medium
  │   ├── conf_encoder.h5
  │   └── password_model.h5
  └── UNCM_medium_8096con_2048pm
      ├── conf_encoder.h5
      └── password_model.h5
  ```

* Parse the input credential file according the "How to parse password leaks" section that follows.

Once everything is ready, you can execute *'compute_guess_numbers.py'*. This takes 3 arguments as input:

1. The configuration file of the pre-trained model you want to use  (i.e., one of the files in *'./configs'*). For instance: *'configs.UNCM_medium_8096con_2048pm'* 
2. A (properly formatted) file containing the credentials for which you want to compute the guess numbers
3. The path  of the output file where to save the computed  guess numbers

A complete example:

```
python compute_guess_numbers.py configs.UNCM_medium_8096con_2048pm examples/findfriendz.com__NOHASH__Social.txt findfriendz.com_guesses.txt
```

The output is a textual file, where each line is a triplet: (*password, probability, guess_number*) separated by '\t'. For instance:

```
asdfgf	1.029932955601332e-05	3936
namasivaya	4.05380477846202e-06	8548
computer005	1.2492302120912832e-08	4406008
promise	1.3209220129046798e-05	2819
banty2354	3.8314557667485784e-11	1090046796
imwburns	7.516712154905258e-12	5427385894
SHONALUVU	2.4051117169236167e-11	1737098939
niranjan	2.7752351285381393e-05	1228
rajivgandhi	1.6550660186788736e-08	3277654
sanjota	1.634404684965732e-08	3307478
sexsixsex	4.0660260013224e-10	128403565
seasons	4.261238639896672e-06	8317
onlyme	4.399283407597418e-05	674
rekhavijay	1.5487582027642613e-07	311469
devil1987	1.7559227302992068e-07	266688
patanahin	1.1765397008351606e-07	408119
...
```

The order in which passwords are listed in the output file is the same as the original file.

### How to parse password leaks:

The *'compute_guess_numbers.py'* only supports a specific input file format. This is a txt file where each row represents  a pair *(email_address, password)*.  Each pair is represented by 4 entries separated by '\t'. Those are:

1. email username
2. email provider (without '@')
3. email top domain (without the initial '.')
4. password

For instance, the entry *(dario.pasquini@gmail.com, password123)* becomes: *"dario.pasquini\tgmail\tcom\tpassword123"*. An example is provided in *examples/fakeleak.txt*. 

---

You  can convert (the more common) credential files of the type:

```
example1@gmail.com:password1
example2@yahoo.com:password2
...
```

into to the format supported by  our code using the script: *'utilities/parse_leak.py'.* 

For instance:

```
python parse_leak.py input.text output.txt ascii
```

The output in *'output.txt'* should look something like this:

````
example1	gmail	com	password1
example2	yahoo	com	password2
...
````

## How to train your own UNCM:

Everything begins with the creation of a configuration file, which should be placed in the directory *'./configs/'*. This config file serves as the blueprint for your model, encompassing various aspects such as architecture, data sources, and training log destinations. A collection of sample configuration files can be found in the *'./configs/'* directory. Additionally, the file *'./configs/\_\_init\_\_.py'* contains the majority of the default configurations. You have the flexibility to import *'\_\_init\_\_.py'* or other templates into your configuration and modify specific parameters according to your requirements.

To illustrate, let's say you need a smaller password model for your UNCM. In this scenario, you can create a new configuration file named *'./configs/UNCM_small.py'* and redefine only the parameters that need to be changed.

```
from UNCM_medium import *

hparams['decoder_arch']['rnn_size'] = 128
```

(If you want to train a UNCM, make sure that *hparams['conditional']* is set to *True*!)

**Once you correctly prepared the training and validation data (see next section⬇️),** you can execute the training script by:

```
mkdir ./logs
python train.py configs.UNCM_small
```

If all processes executed successfully, *TensorBoard* logs and checkpoints will be saved in *'./logs/UNCM_small/'* throughout the training, and the final model will be exported to *'./keras_models/UNCM_small/'*.

###  Training and Validation data:

Training and validation leak files must be placed in the path you assigned to the attribute *'hparams[dataset_dir_home]*' in ./configs/\_\_init\_\_.py.  For instance, you can set it to:

```
hparams = {
    'dataset_dir_home' : "./data/",
 ....
```

Within the path specified by *'hparams[dataset_dir_home]*', two additional directories, namely *'train/'* and *'val/'*, need to be created to accommodate the training and validation leaks.

In these directories, each leak should be stored in a separate .txt file, adhering to the format described in the **"How to parse password leaks" ** section ⬆️. Your directory structure should look something like this: 

```
data
├──train
│ ├── 01186mb.ca__NOHASH__Blogs.txt
│ ├── 012.ca__NOHASH__Business.txt
│ ├── 02asat.photoherald.com__HASH_NOHASH__Business.txt
│ ├── 02grow.ca__HASH_NOHASH__NoCategory.txt
│ ├── 030casting.de__HASH_NOHASH__Entertainment.txt
│ ├── 039.ca__HASH_NOHASH__NoCategory.txt
│ ├── 03designscommunications.ca__NOHASH__NoCategory.txt
│ ├── 03pure.ca__HASH_NOHASH__NoCategory.txt
│ ├── 0420.ca__NOHASH__NoCategory.txt
│ ...
└──val
  ├── 059.ca__HASH_NOHASH__NoCategory.txt
  ├── 07zr.ca__HASH_NOHASH__NoCategory.txt
  ├── 0charges.com__NOHASH__Business.txt
  ├── 1000literaryagents.com__NOHASH__Business.txt
  ├── 100people.org__HASH_NOHASH__Reference.txt
  ├── 112_delft.nl__HASH_NOHASH__Business.txt
  ├── 123php.de__HASH_NOHASH__IT.txt
  ├── 124sold.gr__HASH_NOHASH__Auction.txt
	....
```



🚨 **The training set should be a heterogeneous collection of password leaks coming from different sources** 🚨

In the paper, the models were trained and evaluated on an extensive dataset comprising **11922** distinct password leaks! Training with a smaller dataset would not yield meaningful results.

### Private UNCMs

#### Training 

To train your own UNCM capable of handling differentially private configuration seeds, it is enough to set the following attributes in the model configuration file:

```
hparams['encoder_arch']['mixarch_type'] = 10
hparams['DP_params'] = (1., 3.)
```

*'mixarch_type'=10* tells the model to use the differentially private attention mechanism introduced in the paper.  The two values in *'DP_params'* can be arbitrarily chosen and are the **sensitivity bound on l2 norm** and **noise multiplier** respectively.  An example of configuration file can be found in *'configs/DP_UNCM_medium.py'*.

Once configured the file, you can train the model following the *'How to train your own UNCM'* section.

#### Inference

If you set *'mixarch_type'=10'* during the training, the output of the configuration encoder will automatically become differentially private. This output can then be used just like any other seed. For an illustrative example, refer to '*get_guess_numbers.ipynb*'.  

## Requirements:

* TensorFlow 2.x 
* Python3.(>=6)



