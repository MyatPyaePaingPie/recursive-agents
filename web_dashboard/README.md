# RLM Web Dashboard

**Lightweight web interface for testing and visualizing Recursive Language Models**

## Features

✨ **Test Management**
- View all tests organized by category
- Run individual tests or entire test suite
- See real-time test results

⚡ **Live Execution**
- Watch transparent RLM execution in real-time
- See every event as it happens
- Visualize code generation, validation, and execution
- Track token usage and timing

📊 **Statistics**
- Real-time metrics (tokens, time, calls, events)
- Event logging with timestamps
- Performance monitoring

🎨 **Modern UI**
- Dark theme optimized for code viewing
- Real-time updates via WebSockets
- Responsive design

## Quick Start

### 1. Install Dependencies

```powershell
pip install flask flask-socketio
```

### 2. Run the Dashboard

```powershell
cd web_dashboard
python app.py
```

### 3. Open in Browser

```
http://localhost:5000
```

## Screenshots

### Tests Tab
- Lists all tests from `tests/unit/`, `tests/integration/`, `tests/security/`
- Click any test to run it
- See pass/fail status in real-time
- View test output

### Live Execution Tab
- Enter your query and context
- Choose LLM provider (Ollama, Groq, OpenAI)
- Click "Run Demo" to start
- Watch events stream in real-time:
  - Context loading
  - Code generation
  - Validation
  - Sandbox execution
  - LLM calls
  - Recursion steps
  - Result aggregation

### Examples Tab
- Browse all example scripts
- Quick access to demos

## Usage

### Running Tests

1. Go to **Tests** tab
2. Click "Run All Tests" or click individual tests
3. Watch results appear in real-time
4. View detailed output below

### Watching Execution

1. Go to **Live Execution** tab
2. Enter your query (or use default)
3. Modify context if needed
4. Select provider:
   - **Ollama**: Uses your local Mistral 7B
   - **Groq**: Cloud API (fast, free tier)
   - **OpenAI**: Cloud API (paid)
5. Click **Run Demo**
6. Watch the magic happen! ✨

### Understanding Events

Events show you everything:
- **CONTEXT_LOADING**: Loading and chunking context
- **LLM_THINKING**: Model is generating response
- **CODE_GENERATED**: Code created for processing
- **CODE_VALIDATING**: Checking code safety
- **CODE_EXECUTING**: Running code in sandbox
- **RECURSION_STEP**: Recursive processing happening
- **AGGREGATION**: Combining results

## Architecture

```
web_dashboard/
├── app.py                  # Flask backend with Socket.IO
├── templates/
│   └── index.html          # Frontend UI
└── README.md               # This file
```

**Backend (Flask + Socket.IO):**
- Discovers and runs tests via pytest
- Runs transparent RLM demos
- Streams events to frontend in real-time

**Frontend (Vanilla JS + Socket.IO):**
- No framework overhead - just clean HTML/CSS/JS
- Real-time updates via WebSockets
- Responsive design

## How It Works

### Test Running

1. Dashboard scans `tests/` directory for `test_*.py` files
2. Clicking a test runs: `pytest tests/path/to/test.py -v`
3. Output streams back via Socket.IO
4. Results displayed in real-time

### Transparent Execution

1. User submits query + context
2. Backend creates `TransparentEngine` with callback
3. Callback sends each event to frontend via Socket.IO
4. Frontend displays events as they arrive
5. Final result shown with statistics

## Customization

### Change Port

```python
# In app.py, line at the end:
socketio.run(app, port=8080)  # Change to 8080
```

### Add Custom Views

Edit `templates/index.html` and add a new tab:

```html
<button class="tab" onclick="switchTab('custom')">My View</button>
```

### Style Tweaks

All CSS is in `<style>` tag in `index.html`. Modify colors, sizes, etc.

## Requirements

- Python 3.10+
- Flask
- Flask-SocketIO
- Your RLM package installed (`pip install -e .`)

## Troubleshooting

### "Address already in use"

Port 5000 is taken. Change port:
```python
socketio.run(app, port=5001)
```

### "Module not found"

Install dependencies:
```powershell
pip install flask flask-socketio
```

### Events not showing

1. Check console for errors
2. Make sure Ollama is running: `ollama serve`
3. Verify model is available: `ollama pull mistral:7b`

### Tests not appearing

1. Make sure you're in project root
2. Tests should be in `tests/` directory
3. Test files must start with `test_`

## Advanced

### Run on Network

To access from other devices on your network:

```python
# In app.py:
socketio.run(app, host='0.0.0.0', port=5000)
```

Then access via: `http://YOUR_IP:5000`

### Production Deployment

For production, use proper WSGI server:

```powershell
pip install gunicorn eventlet
gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:5000 app:app
```

## Next Steps

1. **Run it**: `python app.py`
2. **Open browser**: http://localhost:5000
3. **Run tests**: Click "Run All Tests"
4. **Watch execution**: Go to "Live Execution" tab and click "Run Demo"
5. **Enjoy**: See your RLM system in action! 🚀

## Tips

- Keep dashboard open while developing - auto-refreshes test list
- Use Live Execution to debug your RLM queries
- Watch event timings to identify bottlenecks
- Try different providers to compare performance

---

**Built for the RLM project - Making AI transparency beautiful! ✨**

