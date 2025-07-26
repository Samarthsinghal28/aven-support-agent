#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Set environment variables from .env file
export $(grep -v '^#' .env | xargs)

# Start the server with Gunicorn
# Using 4 worker processes and binding to port 8000
# The app is specified as server:app (module:FastAPI instance)
echo "Starting Aven Support AI server with Gunicorn..."
gunicorn server:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120 --access-logfile - --error-logfile -

# Note: You may need to adjust the number of workers based on your server's resources
# A common formula is (2 x number_of_cores) + 1 