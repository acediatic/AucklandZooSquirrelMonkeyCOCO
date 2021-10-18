# MASK_RCNN Auckland Zoo Squirrel Monkeys - PB1

## Setup

1. Clone this repository to your local device
2. Create a virtual environment running <code>Python 3.7</code> by running <code> python -m venv ./venv</code>
3. Activate the virtual environment by running <code>& ./venv/Scripts/Activate.ps1</code> (for Windows)
4. Clone [Mask_RCNN](https://github.com/acediatic/MaskRCNN_TF1) into the project folder (<code>git submodule init git@github.com:acediatic/MaskRCNN_TF1.git</code>)
5. (If wanting to use GPU): install CUDA and cuDNN versions as specificed in [Mask_RCNN](https://github.com/acediatic/MaskRCNN_TF1)
6. <code>cd</code> into the new Mask_RCNN folder (rename to <code>Mask_RCNN</code> if necessary) 
7. Run <code>pip install -r requirements.txt</code> to load the requirements to pip for this repo
8. Run <code>pip install .</code> to install the MaskRCNN library into pip
9. Install iPyKernel by running <code>python -m ipykernel install</code>
10. Train away!
