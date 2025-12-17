Acme Security Bot

1. Set `OPENAI_API_KEY` in `.env`
2. Run `python initialize.py`: loads 1,000 historical tickets, generates embeddings
3. Run `python main.py` starts server on port 8000
4. POST to `/threads` to create conversation, then POST to `/threads/{id}/messages` with `{"text": "your request"}`
5. Bot classifies request, extracts fields, asks follow-ups for missing info, and makes Approved/Rejected decisions based on historical patterns
6. GET `/health` for 30-day risk analysis with pattern detection