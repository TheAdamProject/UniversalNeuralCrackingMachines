



# 🌎 Universal Neural-Cracking-Machines (UNCMs)
Code and pre-trained models for the paper: *"Universal Neural-Cracking-Machines: Self-Configurable Password Models from Auxiliary Data"*  [🎉IEEE S&P'24🎉]  by [Dario Pasquini](https://pasquini-dario.github.io/me/), [Giuseppe Ateniese](https://ateniese.github.io), and [Carmela Troncoso](http://carmelatroncoso.com).

🔥 **All  the main functionalities are now available ** 🔥, but we are planning to add more stuff:

#### Long term goals:

- [ ] Adding improved models trained on (hard) synthetic password leaks. 
- [ ] Adding compressed and quantized pre-trained models for browser inference.
- [ ] Adding JavaScript interface. 
- [ ] Adding bigger pre-trained UNCMs for server-side estimation.

## Download pre-trained models: 
Currently, we offer three models aimed at reproducing the results in the paper:

* [UNCM_medium_8096con_2048pm](https://drive.google.com/drive/folders/1Xf549jF6zo2zlZ4ZbfH3cxN_kEpSZm4K?usp=share_link): The standard UNCM.
* [Baseline](https://drive.google.com/drive/folders/19u2Ld3PWIvRZ9ejYCVD9dcoGsl6e8pKN?usp=share_link): Baseline model (non-conditional password model)
* [DP-UNCM](https://drive.google.com/drive/folders/1wWi71UJrcObwoBt9GkH0Q-nbnDan_M4Y?usp=share_link): UNCM trained to handle DP configuration seeds. 

## How to use UNCMs:

We have 3 playground notebooks: 

* *sampling_example.ipynb*: Notebook on how to sample passwords from the models (not meant for password guessing).
* *get_probability.ipynb*: Notebook on how to assign probabilities to plaintext passwords with a UNCM. 
* *get_guess_numbers.ipynb*: Notebook on how to assign guess numbers to plaintext passwords with a UNCM. 

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

## Requirements:

* TensorFlow 2.x 
* Python3.(>=6)



