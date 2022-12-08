### Calcium extraction with DeepWonder

**Setting up the environment**
* Create a new Python 3.9 virtual environment named `DWonder` in Anaconda. Activate 'DWonder' environment.
* Install GPU-version pytorch (Stable build) from pytorch website: https://pytorch.org/get-started/locally/
  *  Make sure pytorch is installed successfully by running the commands listed below in python.

```
conda activate DWonder
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python
>>> import torch
>>> torch.cuda.is_available() # should be true
```

**Process the data**
We provide a one-click running script for running DeepWonder. To process your data:
* Place the `.tif` calcium video recorded by SOMM in '\Experiments\Your_Experiment_Index' folder.
* Modify the line $Indexes = 'Your_Experiment_Index' in 'extract.ps1' 
* Run the script with Windows Powershell.