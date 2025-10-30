# app.py

from flask import Flask
import os

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Import and register routes
    from routes import register_routes
    register_routes(app)
    
    return app

# Create app instance for gunicorn
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)