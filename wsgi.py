import sys

project_home = '/path/to/your/project'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from app import app as application