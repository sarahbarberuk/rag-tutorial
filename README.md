# rag-tutorial

## Setup

1. Create .env file and copy .env.example into it. Add your keys.
2. Create a virtual environment, and activate it:

`python -m venv venv
source venv/bin/activate`

3. Install dependencies:
   `pip install -r requirements.txt`

## Ingest

How to run this

`python ingest.py`

## Checking the vector DB

Run this script to take a quick look at a sample of what is now in the vector DB. To check it worked, and for cuuriosity.

`python check_vectordb.py`

## Querying the vector DB

`python query.py`

Ask questions like:

- "Which GPU should I buy for machine learning?"
- "Is the Ryzen 9 9950X compatible with DDR4 RAM?"
- "What is the warranty on the weather forecast?" - tests the "I don't know" behavior
- "What is the returns policy?" - tests when happesn when there are conflicting documents from different dates
