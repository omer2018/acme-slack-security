import json
import numpy as np
from database import HistoricalTicket
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_openai_client, get_db_session

def _call_llm_for_text(prompt):
    openai_client = get_openai_client()
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def _call_llm_for_json(prompt):
    openai_client = get_openai_client()
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def find_similar_historical_tickets(query_text, number_of_tickets_to_retrieve=5):
    openai_client = get_openai_client()
    query_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    ).data[0].embedding
    
    with get_db_session() as database_session:
        all_historical_tickets = database_session.query(HistoricalTicket).all()
        
        historical_ticket_embeddings = [json.loads(ticket.embedding) for ticket in all_historical_tickets]
        similarity_scores = cosine_similarity([query_embedding], historical_ticket_embeddings)[0]
        
        top_similar_indices = similarity_scores.argsort()[-number_of_tickets_to_retrieve:][::-1]
        most_similar_tickets = [all_historical_tickets[index] for index in top_similar_indices]
        
        tickets_data = [
            {
                'request_summary': ticket.request_summary,
                'request_type': ticket.request_type,
                'fields_provided': ticket.fields_provided,
                'security_risk_score': ticket.security_risk_score,
                'outcome': ticket.outcome
            }
            for ticket in most_similar_tickets
        ]
        
        return tickets_data

def classify_security_request(user_message, similar_historical_tickets=None):
    if similar_historical_tickets is None:
        similar_historical_tickets = find_similar_historical_tickets(user_message, number_of_tickets_to_retrieve=3)
    
    # Get all unique request_type from the database
    with get_db_session() as database_session:
        unique_request_types = database_session.query(HistoricalTicket.request_type).distinct().all()
        request_types_list = sorted([rt[0] for rt in unique_request_types])
        request_types_string = ", ".join(request_types_list)
    
    formatted_examples = "\n".join([
        f"- '{ticket['request_summary']}' : {ticket['request_type']}"
        for ticket in similar_historical_tickets[:3]
    ])

    print(formatted_examples)
    
    classification_prompt = f"""Based on these examples:
                                {formatted_examples}

                                Classify this request into one of: {request_types_string}

                                Request: {user_message}

                                Reply with just the request type."""
    
    return _call_llm_for_text(classification_prompt)

def extract_required_fields_from_request(user_message, request_type):
    with get_db_session() as database_session:
        sample_ticket_for_request_type = database_session.query(HistoricalTicket).filter_by(request_type=request_type).first()
        mandatory_fields_list = sample_ticket_for_request_type.mandatory_fields.split("; ") if sample_ticket_for_request_type else []
    
    extraction_prompt = f"""Extract information from this security request.

                            Request: {user_message}

                            Required fields: {', '.join(mandatory_fields_list)}

                            IMPORTANT: Also extract what specific permission, resource, or access is being requested (e.g., "AWS admin access", "GitHub repository access", "VPN access to production network").

                            For each field, return either the value found or "MISSING".

                            Reply in JSON format:
                            {{
                            "field_name": "value or MISSING",
                            "requested_access": "specific access requested"
                            }}"""
    
    extracted_fields_data = _call_llm_for_json(extraction_prompt)
    
    requested_access = extracted_fields_data.pop("requested_access", None)
    if requested_access and requested_access != "MISSING":
        extracted_fields_data["Requested Access"] = requested_access
    
    provided_fields = {field_name: field_value for field_name, field_value in extracted_fields_data.items() if field_value != "MISSING"}
    missing_fields = [field_name for field_name, field_value in extracted_fields_data.items() if field_value == "MISSING"]
    
    return provided_fields, missing_fields, mandatory_fields_list

def generate_follow_up_question(missing_field_name, conversation_context=""):
    question_prompt = f"""Generate a natural, friendly Slack message asking for: {missing_field_name}

                        Context: {conversation_context}

                        Keep it brief and conversational. One sentence. Use the person's actual name if provided."""
                            
    return _call_llm_for_text(question_prompt)

def generate_follow_up_questions(missing_fields_list, conversation_context):
    question_prompt = f"""Generate a natural, friendly Slack message asking for these missing fields: {', '.join(missing_fields_list)}

                        Context: {conversation_context}

                        Keep it conversational and ask for all fields in one message. Use the person's name if provided. 
                        No new lines in the message. No emojis."""
    
    return _call_llm_for_text(question_prompt)

def make_security_decision(user_message, request_type, provided_fields, missing_fields, similar_historical_tickets=None):
    if missing_fields:
        return "Info Requested", "Missing required fields", None, None
    
    if similar_historical_tickets is None:
        similar_historical_tickets = find_similar_historical_tickets(user_message, number_of_tickets_to_retrieve=5)
    
    historical_cases_context = "\n".join([
        f"Similar case: {ticket['request_summary']}\n"
        f"Fields: {ticket['fields_provided']}\n"
        f"Risk: {ticket['security_risk_score']}\n"
        f"Outcome: {ticket['outcome']}\n"
        for ticket in similar_historical_tickets
    ])
    
    decision_prompt = f"""You are Acme's security decision engine.

                Historical similar cases:
                {historical_cases_context}

                Current request:
                Type: {request_type}
                Details: {user_message}
                Fields provided: {', '.join(provided_fields.keys())}

                Based on Acme's historical practice, should this be Approved or Rejected?

                Reply in JSON:
                {{
                "decision": "Approved or Rejected",
                "rationale": "brief explanation",
                "risk_score": 0-100,
                "confidence_score": 0.0-1.0 (how confident are you in this decision based on similarity to historical cases)
                }}"""
                
    decision_result = _call_llm_for_json(decision_prompt)
    
    return decision_result["decision"], decision_result["rationale"], decision_result["risk_score"], decision_result["confidence_score"]
