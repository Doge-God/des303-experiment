# Set the OLLAMA_HOST environment variable for phi4 on port 11434.
$env:OLLAMA_HOST = "http://localhost:11434"

Write-Host "phi4 mini, 11434"
# Run the phi4 model.
ollama run phi4

# Keep the window open so you can view output or errors.
Write-Host "phi4 model is running. Press Enter to exit..."
Read-Host
