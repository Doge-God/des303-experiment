# Set the OLLAMA_HOST environment variable for qwen2.5:1.5b on port 11436.
$env:OLLAMA_HOST = "http://localhost:11436"

Write-Host "qwen2.5 1.5b, 11436"
# Run the qwen2.5:1.5b model.
ollama run "qwen2.5:1.5b"

# Keep the window open so you can view output or errors.
Write-Host "qwen2.5:1.5b model is running. Press Enter to exit..."
Read-Host
