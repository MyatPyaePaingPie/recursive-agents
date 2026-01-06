# Installing Ollama on Windows (Manual Method)

## Method 1: Direct Download (Recommended)

Since winget is having issues, download directly from Ollama's website.

### Steps:

1. **Open your browser and go to:**
   ```
   https://ollama.com/download
   ```

2. **Click "Download for Windows"**
   - This will download `OllamaSetup.exe` (~200MB)

3. **Run the installer**
   - Double-click `OllamaSetup.exe`
   - Follow the installation wizard
   - Accept defaults (installs to `C:\Users\<your-user>\AppData\Local\Programs\Ollama\`)

4. **Verify installation**
   - Open a **NEW** PowerShell window (important!)
   - Run:
   ```powershell
   ollama --version
   ```

5. **If still not recognized, restart PowerShell or add to PATH manually:**
   ```powershell
   $env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"
   ```

---

## Method 2: PowerShell Direct Download

If you want to automate the download:

```powershell
# Download installer
$installerUrl = "https://ollama.com/download/OllamaSetup.exe"
$installerPath = "$env:USERPROFILE\Downloads\OllamaSetup.exe"

Write-Host "Downloading Ollama installer..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath

Write-Host "Download complete! Please run: $installerPath" -ForegroundColor Green
Write-Host "After installation, restart PowerShell and run: ollama --version" -ForegroundColor Yellow

# Open downloads folder
Start-Process explorer.exe "$env:USERPROFILE\Downloads"
```

Then manually run the installer from your Downloads folder.

---

## Method 3: Alternative - LM Studio (No Installation Issues)

If Ollama continues to have problems, use **LM Studio** instead:

1. **Download LM Studio:**
   ```
   https://lmstudio.ai/
   ```

2. **Install it** (standard Windows installer)

3. **Download a model:**
   - Open LM Studio
   - Click "Search" tab
   - Search "Mistral 7B"
   - Click "Download" on `bartowski/Mistral-7B-Instruct-v0.3-GGUF`
   - Select `Q5_K_M` variant

4. **Start local server:**
   - Click "Local Server" tab
   - Click "Start Server"
   - Server runs on `http://localhost:1234`

5. **Use with RLM:**
   - Set in `.env`: `LOCAL_MODEL_TYPE=lm-studio`
   - Base URL: `http://localhost:1234/v1`

---

## Troubleshooting

### "ollama not recognized" after installation

**Solution 1: Restart PowerShell**
Close all PowerShell windows and open a new one.

**Solution 2: Check installation location**
```powershell
# Check if Ollama is installed
Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
```

**Solution 3: Add to PATH manually**
```powershell
# Temporary (current session)
$env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"

# Permanent (add to system PATH)
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "User") + ";$env:LOCALAPPDATA\Programs\Ollama",
    "User"
)
```

**Solution 4: Run with full path**
```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --version
```

### "Access denied" or permission errors

**Solution**: Run PowerShell as Administrator
- Right-click PowerShell → "Run as Administrator"
- Try installation again

### Download is very slow

**Solution**: Use a download manager or try during off-peak hours
- Or use LM Studio as alternative (Method 3)

---

## After Successful Installation

Once Ollama is installed and recognized:

```powershell
# Verify it works
ollama --version

# Pull your first model (Mistral 7B)
ollama pull mistral:7b

# Test it
ollama run mistral:7b "Write a hello world in Python"

# Exit chat
# Type /bye or press Ctrl+D
```

---

## Next Steps

After Ollama is working:
1. Pull models: `ollama pull mistral:7b`
2. Continue with `QUICKSTART_LOCAL.md` Step 3
3. Set up Python environment
4. Configure `.env` for local models

---

## Getting Help

If you're still stuck:
1. Check Ollama docs: https://github.com/ollama/ollama/blob/main/docs/windows.md
2. Try LM Studio instead (easier installation)
3. Or use Groq API (cloud, free tier): https://console.groq.com/


