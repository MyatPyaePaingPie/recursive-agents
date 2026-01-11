"""Simple Flask dashboard for RLM testing and visualization.

This provides a web UI to:
- View and run all tests
- Watch transparent execution in real-time
- Visualize recursion trees
- See event logs
- Support lightweight/heavyweight model tiers

Run:
    pip install flask flask-socketio python-dotenv
    python web_dashboard/app.py

    Then open: http://localhost:5000
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from threading import Thread

# Load .env file from project root
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    print(f"[DEBUG] Loading .env from {env_file}")
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print("[DEBUG] .env loaded successfully")
    except ImportError:
        print("[DEBUG] python-dotenv not installed, loading .env manually")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
                    print(f"[DEBUG] Set {key.strip()}=***")
else:
    print(f"[DEBUG] No .env file found at {env_file}")

# Print API key status
print(f"[DEBUG] GROQ_API_KEY present: {bool(os.environ.get('GROQ_API_KEY'))}")
print(f"[DEBUG] OPENAI_API_KEY present: {bool(os.environ.get('OPENAI_API_KEY'))}")
print(f"[DEBUG] ANTHROPIC_API_KEY present: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

# Add src directory to path so we can import rlm
sys.path.insert(0, str(project_root / "src"))

from rlm import RLMConfig
from rlm.core.transparent import Event, EventType, TransparentEngine
from rlm.models import create_llm, ModelTiers, ModelTier

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rlm-dashboard-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
current_events = []
test_results = {}


def find_all_tests():
    """Find all test files in the tests directory."""
    tests_dir = Path(__file__).parent.parent / "tests"
    test_files = []
    
    for test_file in tests_dir.rglob("test_*.py"):
        rel_path = test_file.relative_to(tests_dir.parent)
        test_files.append({
            'path': str(rel_path),
            'name': test_file.stem,
            'category': test_file.parent.name,
        })
    
    return test_files


def run_pytest(test_path):
    """Run pytest on a specific test file."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    return {
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode,
        'passed': result.returncode == 0
    }


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/tests')
def get_tests():
    """Get list of all tests."""
    tests = find_all_tests()
    return jsonify({'tests': tests})


@app.route('/api/tests/run', methods=['POST'])
def run_test():
    """Run a specific test."""
    data = request.json
    test_path = data.get('test_path')
    
    if not test_path:
        return jsonify({'error': 'No test path provided'}), 400
    
    # Run in background and emit results via socketio
    def run_async():
        result = run_pytest(test_path)
        test_results[test_path] = result
        socketio.emit('test_result', {
            'test_path': test_path,
            **result
        })
    
    Thread(target=run_async).start()
    
    return jsonify({'status': 'running', 'test_path': test_path})


@app.route('/api/tests/run-all', methods=['POST'])
def run_all_tests():
    """Run all tests."""
    def run_async():
        result = run_pytest("tests/")
        socketio.emit('all_tests_result', result)
    
    Thread(target=run_async).start()
    
    return jsonify({'status': 'running'})


@app.route('/api/examples')
def get_examples():
    """Get list of all examples."""
    examples_dir = Path(__file__).parent.parent / "examples"
    examples = []

    for example_file in examples_dir.glob("*.py"):
        if example_file.name.startswith('_'):
            continue
        examples.append({
            'name': example_file.stem,
            'path': str(example_file.relative_to(examples_dir.parent))
        })

    return jsonify({'examples': examples})


@app.route('/api/model-tiers')
def get_model_tiers():
    """Get available model tiers for each provider."""
    from rlm.models.tiers import MODEL_CATALOG, ModelTier

    tiers_info = {}
    for provider, specs in MODEL_CATALOG.items():
        tiers_info[provider] = {}
        for tier in ModelTier:
            spec = specs[tier]
            tiers_info[provider][tier.value] = {
                'model': spec.model,
                'description': spec.description,
                'context_window': spec.context_window,
                'cost_per_1k_tokens': spec.cost_per_1k_tokens,
                'tokens_per_second': spec.tokens_per_second,
            }

    return jsonify({'tiers': tiers_info, 'providers': list(MODEL_CATALOG.keys())})


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {'data': 'Connected to RLM Dashboard'})
    print(f"Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")


@socketio.on('run_transparent_demo')
def handle_transparent_demo(data):
    """Run a transparent demo and stream events."""
    global current_events
    current_events = []

    provider = data.get('provider', 'ollama')
    query = data.get('query', 'What is recursion?')
    context = data.get('context', 'Recursion is a programming technique...')
    use_tiers = data.get('use_tiers', True)

    print(f"\n{'='*60}")
    print(f"[DEBUG] run_transparent_demo called")
    print(f"[DEBUG] Provider: {provider}, Use tiers: {use_tiers}")
    print(f"[DEBUG] Query: {query[:50]}...")
    print(f"{'='*60}\n")

    def event_callback(event: Event):
        """Callback that sends events to the client."""
        print(f"[EVENT] {event.type.value} (depth={event.depth})")

        # Serialize event data properly, handling special cases
        serialized_data = {}
        for key, value in event.data.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serialized_data[key] = value
            elif isinstance(value, list):
                serialized_data[key] = value[:10]  # Limit list size
            else:
                serialized_data[key] = str(value)

        event_data = {
            'type': event.type.value,
            'timestamp': event.timestamp.isoformat(),
            'depth': event.depth,
            'data': serialized_data,
            'duration_ms': event.duration_ms,
            # Add category for color coding
            'category': get_event_category(event.type),
        }
        current_events.append(event_data)
        # Use socketio.emit with namespace for background thread compatibility
        socketio.emit('rlm_event', event_data, namespace='/')

    def run_sync():
        """Run the demo synchronously to avoid async/thread issues."""
        try:
            print("[DEBUG] Starting demo execution...")
            config = RLMConfig()
            print(f"[DEBUG] Config loaded")

            # Use model tiers if requested
            if use_tiers:
                try:
                    print(f"[DEBUG] Creating model tiers for {provider}...")
                    tiers = ModelTiers(provider)
                    code_llm = tiers.get_lightweight()
                    reasoning_llm = tiers.get_heavyweight()
                    print(f"[DEBUG] Code LLM: {code_llm.model}")
                    print(f"[DEBUG] Reasoning LLM: {reasoning_llm.model}")

                    # Emit model info
                    socketio.emit('model_info', {
                        'code_model': tiers.get_spec(ModelTier.LIGHTWEIGHT).model,
                        'code_description': tiers.get_spec(ModelTier.LIGHTWEIGHT).description,
                        'reasoning_model': tiers.get_spec(ModelTier.HEAVYWEIGHT).model,
                        'reasoning_description': tiers.get_spec(ModelTier.HEAVYWEIGHT).description,
                        'use_tiers': True,
                    }, namespace='/')
                except Exception as e:
                    print(f"[DEBUG] Tier creation failed: {e}, falling back to single model")
                    # Fallback to single model
                    code_llm = create_llm(provider=provider)
                    reasoning_llm = code_llm
                    socketio.emit('model_info', {
                        'code_model': code_llm.model,
                        'reasoning_model': code_llm.model,
                        'use_tiers': False,
                        'fallback_reason': str(e),
                    }, namespace='/')
            else:
                print(f"[DEBUG] Creating single model for {provider}...")
                llm = create_llm(provider=provider)
                code_llm = llm
                reasoning_llm = llm
                print(f"[DEBUG] LLM: {llm.model}")
                socketio.emit('model_info', {
                    'code_model': llm.model,
                    'reasoning_model': llm.model,
                    'use_tiers': False,
                }, namespace='/')

            print("[DEBUG] Creating TransparentEngine...")
            engine = TransparentEngine(
                llm=reasoning_llm,
                code_llm=code_llm,
                config=config,
                callback=event_callback
            )

            print("[DEBUG] Starting engine.process()...")
            # Run the async process in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(engine.process(query=query, context=context))
            finally:
                loop.close()

            print(f"[DEBUG] Processing complete! Answer length: {len(result.answer)}")

            # Get event summary
            summary = engine.get_event_summary()

            socketio.emit('demo_complete', {
                'answer': result.answer,
                'total_tokens': result.total_tokens,
                'execution_time': result.execution_time,
                'num_recursive_calls': result.num_recursive_calls,
                'max_depth_reached': result.max_depth_reached,
                'events': current_events,
                'event_summary': summary,
            }, namespace='/')
            print("[DEBUG] demo_complete emitted")

        except Exception as e:
            import traceback
            error_tb = traceback.format_exc()
            print(f"[ERROR] Demo failed: {e}")
            print(f"[ERROR] Traceback:\n{error_tb}")
            socketio.emit('demo_error', {
                'error': str(e),
                'traceback': error_tb
            }, namespace='/')

    # Run in background thread
    print("[DEBUG] Starting background thread...")
    Thread(target=run_sync, daemon=True).start()

    emit('demo_started', {'status': 'running', 'provider': provider, 'use_tiers': use_tiers})


def get_event_category(event_type: EventType) -> str:
    """Get the category for an event type for color coding."""
    category_map = {
        EventType.PROCESS_START: 'lifecycle',
        EventType.PROCESS_END: 'lifecycle',
        EventType.CONTEXT_LOADING: 'context',
        EventType.CONTEXT_LOADED: 'context',
        EventType.CONTEXT_CHUNKED: 'context',
        EventType.LLM_PROMPT_PREPARED: 'llm',
        EventType.LLM_THINKING: 'llm',
        EventType.LLM_RESPONSE_RECEIVED: 'llm',
        EventType.CODE_GENERATED: 'code',
        EventType.CODE_VALIDATING: 'code',
        EventType.CODE_VALIDATION_RESULT: 'code',
        EventType.CODE_EXECUTING: 'execution',
        EventType.CODE_EXECUTION_RESULT: 'execution',
        EventType.RECURSION_START: 'recursion',
        EventType.RECURSION_STEP: 'recursion',
        EventType.RECURSION_SUBMODEL_CALL: 'recursion',
        EventType.RECURSION_END: 'recursion',
        EventType.AGGREGATION_START: 'aggregation',
        EventType.AGGREGATION_END: 'aggregation',
        EventType.ERROR: 'error',
        EventType.FALLBACK_TRIGGERED: 'error',
    }
    return category_map.get(event_type, 'other')


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 RLM Dashboard Starting")
    print("=" * 70)
    print()
    print("Dashboard will be available at: http://localhost:5000")
    print()
    print("Features:")
    print("  - View and run all tests")
    print("  - Watch transparent RLM execution")
    print("  - Visualize recursion trees")
    print("  - See real-time event logs")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

