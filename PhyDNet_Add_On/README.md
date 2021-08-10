## Modified PhyDNet Instruction

Get original code: `git clone` [[PhyDNet repo]](https://github.com/vincent-leguen/PhyDNet) 

Install requirements: `pip install -r requirements.txt`

Replace the `main.py` and add in the `config.yml` into the same directory

Replace the `data` folder from the original PhyDNet with the `data` folder from this repo. 

Set the parameters in the `config.yml`

Run code: `python main.py --config_file config.yml`

Their CVPR 2020 paper "Disentangling Physical Dynamics from Unknown Factors for Unsupervised Video Prediction": https://arxiv.org/abs/2003.01460. If used, please remember to cite them too.
