import sys
import os

# Add the project root (parent of api/) to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from api.main import app
