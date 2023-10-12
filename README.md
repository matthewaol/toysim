# x-ray diffraction data simulator

to use,
clone the repository `git clone <url>`
then
1. create a directory for your developer environment `cd ~` `mkdir dials` `cd dials`
2. get the build script `wget https://raw.githubusercontent.com/dials/dials/main/installer/bootstrap.py`
or navigate to https://dials.github.io/documentation/installation_developer.html
3. run the build script `python boostrap.py`
4. run the environment `export DIALS=~/dials/modules` `source ~/dials/dials`
or use a shell script to run commands
ex. this is my setup ```alias ls="ls --color" 
alias ipython="python -m IPython" 
alias jupyter-notebook="python -m jupyter notebook" 
export DIALS=~/dials/modules
export PS1="[\w]\n$ " 

source ~/dials/dials```
5. after setting up environment, change directory to where your repostiry is `cd ~/toysim`
6. run this command for a sample protein structure: `iotbx.fetch_pdb 4bs7` 
7. start simulating an image `python simulation.py` 
