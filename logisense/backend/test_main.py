import sys
sys.path.insert(0, '.')
import traceback
import app.services.route_optimizer as ro
try:
    with open('app/services/route_optimizer.py', 'r') as f:
        code = f.read()
    exec(code)
except Exception as e:
    traceback.print_exc()
