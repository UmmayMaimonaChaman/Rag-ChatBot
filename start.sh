#!/bin/bash

# Start FastAPI in the background
python backend/main.py &

# Start Streamlit on port 7860
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0
