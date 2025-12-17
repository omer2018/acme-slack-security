import pandas as pd
import json
from database import initialize_database, HistoricalTicket
from datetime import datetime
from utils import get_openai_client, get_db_session

def generate_embeddings_for_texts(texts_list, batch_size=100):
    openai_client = get_openai_client()
    all_generated_embeddings = []
    for start_index in range(0, len(texts_list), batch_size):
        current_batch = texts_list[start_index:start_index+batch_size]
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=current_batch
        )
        all_generated_embeddings.extend([embedding_item.embedding for embedding_item in embedding_response.data])
    return all_generated_embeddings

def load_historical_tickets_from_csv(csv_file_path="data/acme_security_tickets.csv"):
    print("Loading CSV")
    tickets_data_frame = pd.read_csv(csv_file_path)
    
    texts_for_embedding = [
        f"{ticket_row['request_type']}: {ticket_row['request_summary']}\n{ticket_row['details']}"
        for _, ticket_row in tickets_data_frame.iterrows()
    ]
    
    print(f"Generating embeddings")
    generated_embeddings = generate_embeddings_for_texts(texts_for_embedding)
    
    print("Saving to database")
    with get_db_session() as database_session:
        for ticket_index, ticket_row in tickets_data_frame.iterrows():
            historical_ticket_record = HistoricalTicket(
                ticket_id=ticket_row['ticket_id'],
                request_type=ticket_row['request_type'],
                request_summary=ticket_row['request_summary'],
                details=ticket_row['details'],
                mandatory_fields=ticket_row['mandatory_fields'],
                fields_provided=ticket_row['fields_provided'],
                outcome=ticket_row['outcome'],
                security_risk_score=int(ticket_row['security_risk_score']),
                embedding=json.dumps(generated_embeddings[ticket_index]),
                created_at=datetime.strptime(ticket_row['created_at'], "%Y-%m-%d %H:%M:%S"),
                requester_department=ticket_row['requester_department'],
                requester_title=ticket_row['requester_title']
            )
            database_session.add(historical_ticket_record)
    print(f"Loaded historical tickets with embeddings")

def initialize_application_database():
    initialize_database()
    print("Tables created")
    
    print("Loading historical tickets")
    load_historical_tickets_from_csv()

if __name__ == "__main__":
    initialize_application_database()