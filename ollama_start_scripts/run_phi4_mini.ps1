# Set the OLLAMA_HOST environment variable for phi4-mini on port 11435.
$env:OLLAMA_HOST = "http://localhost:11435"

Write-Host "phi4 mini, 11435"
# Run the phi4-mini model.
ollama run phi4-mini

# Keep the window open so you can view output or errors.

Write-Host "phi4-mini model is running. Press Enter to exit..."
Read-Host
