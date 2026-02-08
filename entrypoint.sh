#!/bin/sh
set -e

echo "Running training..."
cd notebook
python ML_Project_HannanButt.py

echo "Starting Streamlit app..."
cd /app
streamlit run app.py --server.port=$PORT 
