from app import app  # Import the app instance from app.py

# Required by Vercel to run Flask
def handler(request):
    return app(request.environ, start_response=None)
