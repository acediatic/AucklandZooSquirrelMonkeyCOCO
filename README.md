# MASK_RCNN Auckland Zoo Squirrel Monkeys - PB1

## Setup

1. Clone this repository to your local device
2. Using Conda, create an environment running Python 3.7 <code>conda create --name "IndividualIdentification" python=3.7</code> 
3. Activate the virtual environment
4. Install CUDA toolkit and cuDNN using <code>conda install cudatoolkit=11.0</code> and then <code>conda install cudnn=8.0</code>
4. Clone [Mask-RCNN-TF=2](https://github.com/ahmedfgad/Mask-RCNN-TF2) into the project folder (<code>git submodule init https://github.com/ahmedfgad/Mask-RCNN-TF2.git</code>)
5. Download the submodule by running <code>git submodule update</code>
6. <code>cd</code> into the new Mask_RCNN folder (rename to <code>Mask_RCNN</code> if necessary) 
7. In your environment created above, run <code>pip install -r requirements.txt</code> to load the requirements to pip for this repo
8. To replace Tensorflow, run <code>pip uninstall tensorflow</code> and then <code>pip install tensorflow==2.4</code>
8. In your environment created above, install iPyKernel by running <code>python -m ipykernel install</code>
9. Train away!
