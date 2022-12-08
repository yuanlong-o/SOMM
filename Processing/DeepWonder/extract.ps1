Write-Host '==============NEW SESSION STARTED============' -ForegroundColor Red

$Indexes = 'Your_Experiment_Index'

conda activate DWonder
Write-Host 'Mini environment activated' -ForegroundColor Green

foreach ($Index in $Indexes)
{
Write-Host 'Processing experiment: ' + $Index -ForegroundColor Yellow

Write-Host 'Calcuim extracting: removing background...' -ForegroundColor Green
python ca_rmbg.py --index $index

Write-Host 'Calcuim extracting: perform segmentation...' -ForegroundColor Green
python ca_seg.py --i $index
}

Read-Host -Prompt "Press any key to close the terminal"

