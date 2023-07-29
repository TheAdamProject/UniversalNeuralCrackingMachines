



# Universal Neural-Cracking-Machines (UNCMs)
Code and pre-trained models for the paper: *"Universal Neural-Cracking-Machines: Self-Configurable Password Models from Auxiliary Data"*  [🎉IEEE S&P'24🎉]  by [Dario Pasquini](https://pasquini-dario.github.io/me/), [Giuseppe Ateniese](https://ateniese.github.io), and [Carmela Troncoso](http://carmelatroncoso.com).

🔥 **80% of the basic functionalities is available. Work in progress...** 🔥

## Pre-trained models: 
Currently, we offer three models aimed at reproducing the results in the paper:

* [UNCM_medium_8096con_2048pm](https://drive.google.com/drive/folders/1Xf549jF6zo2zlZ4ZbfH3cxN_kEpSZm4K?usp=share_link): The standard UNCM.
* [Baseline](https://drive.google.com/drive/folders/19u2Ld3PWIvRZ9ejYCVD9dcoGsl6e8pKN?usp=share_link): Baseline model (non-conditional password model)
* [DP-UNCM](https://drive.google.com/drive/folders/1wWi71UJrcObwoBt9GkH0Q-nbnDan_M4Y?usp=share_link): UNCM trained to handle DP configuration seeds. 

Documentation on how to train, use, and evaluate the models (i.e., how to use *train.py* and *test.py*) and work with DP-UNCM **will be added within the next weeks**. 

For now, we have: 

* *sampling_example.ipynb*: Notebook on how to sample passwords from the models (not meant for password guessing).
* *get_probability.ipynb*: Notebook on how to assign probabilities to plaintext passwords with a UNCM. 
* *get_guess_numbers.ipynb*: Notebook on how to assign guess numbers to plaintext passwords with a UNCM. 

## Requirements:

* TensorFlow 2.x

## Long term goals:

- [ ]  Adding improved models trained on (hard) synthetic password leaks. 
- [ ]  Adding compressed and quantized pre-trained models for browser inference.
- [ ]   Adding JavaScript interface. 
- [ ]  Adding bigger pre-trained UNCMs for server-side estimation.

