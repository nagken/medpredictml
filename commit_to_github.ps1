Write-Host " Navigating to project directory..."
Set-Location -Path "C:\projects\MedPredictML"

Write-Host " Initializing Git repository..."
git init

Write-Host "Creating README.md..."
Set-Content -Path "README.md" -Value "# MedPredictML"

Write-Host "Adding all files to Git..."
git add .

Write-Host " Committing changes..."
git commit -m "Initial commit"

Write-Host " Setting branch to main..."
git branch -M main

Write-Host " Adding remote repository..."
git remote add origin https://github.com/nagken/medpredictml.git

Write-Host "Pushing to GitHub..."
git push -u origin main

Write-Host " Git commit & push completed successfully!"
