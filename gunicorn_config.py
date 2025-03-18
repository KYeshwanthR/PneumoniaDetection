import os

workers = 4
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
timeout = 120
