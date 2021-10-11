# MASK_RCNN Auckland Zoo Squirrel Monkeys - PB1

## Setup

1. Clone this repository to your local device
2. Create a virtual environment running <code>Python 3.7</code> by running <code> python -m venv ./venv</code>
3. Activate the virtual environment by running <code>& ./venv/Scripts/Activate.ps1</code> (for Windows)
4. Clone [Mask_RCNN](https://github.com/acediatic/MaskRCNN_TF1) into the project folder (<code>git clone git@github.com:acediatic/MaskRCNN_TF1.git</code>)
5. <code>cd</code> into the new Mask_RCNN folder (rename to <code>Mask_RCNN</code> if necessary) 
6. Run <code>pip install -r requirements.txt</code> to load the requirements to pip for this repo
7. Run <code>pip install .</code> to install the MaskRCNN library into pip
8. Install iPyKernel by running <code>python -m ipykernel install .</code>
9. Train away!