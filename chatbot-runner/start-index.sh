#!/bin/bash
cd /home/iuvadito/chatbot
source venv/bin/activate
cd proyecto-couchdb
python3 -m http.server 8080
