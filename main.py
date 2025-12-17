from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from database import Thread, Message, Decision, AuditLog
from llm_service import classify_security_request, extract_required_fields_from_request, generate_follow_up_questions, make_security_decision, find_similar_historical_tickets
from utils import get_db_session
from dotenv import load_dotenv
from datetime import datetime, timedelta
from llm_service import _call_llm_for_json

load_dotenv()

app = FastAPI()

class MessageInput(BaseModel):
    text: str

@app.post("/threads")
def create_new_thread():
    unique_thread_identifier = str(uuid.uuid4())
    
    with get_db_session() as database_session:
        new_thread = Thread(thread_id=unique_thread_identifier, slack_thread_ts=unique_thread_identifier)
        database_session.add(new_thread)
    
    return {"thread_id": unique_thread_identifier}

def _save_user_message(database_session, thread_id: str, message_text: str):
    user_message = Message(thread_id=thread_id, role="user", text=message_text)
    database_session.add(user_message)
    database_session.commit()

def _build_conversation_history(database_session, thread_id: str) -> str:
    all_messages_in_thread = database_session.query(Message).filter_by(thread_id=thread_id).order_by(Message.timestamp).all()
    complete_conversation_text = "\n".join([f"{message_record.role}: {message_record.text}" for message_record in all_messages_in_thread])
    return complete_conversation_text

def _analyze_security_request(complete_conversation_text: str, similar_historical_tickets):
    identified_request_type = classify_security_request(complete_conversation_text, similar_historical_tickets)
    provided_fields, missing_fields, mandatory_fields = extract_required_fields_from_request(complete_conversation_text, identified_request_type)
    return identified_request_type, provided_fields, missing_fields, mandatory_fields

def _determine_next_action(complete_conversation_text: str, identified_request_type: str, provided_fields, missing_fields, similar_historical_tickets):
    next_question_to_ask = None
    final_decision_outcome = None
    decision_rationale = None
    calculated_risk_score = None
    calculated_confidence_score = None
    
    if missing_fields:
        next_question_to_ask = generate_follow_up_questions(missing_fields, complete_conversation_text)
    else:
        final_decision_outcome, decision_rationale, calculated_risk_score, calculated_confidence_score = make_security_decision(
            complete_conversation_text, identified_request_type, provided_fields, missing_fields, similar_historical_tickets
        )
    
    return next_question_to_ask, final_decision_outcome, decision_rationale, calculated_risk_score, calculated_confidence_score

def _upsert_decision_record(database_session, thread_id: str, identified_request_type: str, 
                            provided_fields, missing_fields, mandatory_fields, 
                            final_decision_outcome, decision_rationale, calculated_risk_score, calculated_confidence_score):
    existing_decision_record = database_session.query(Decision).filter_by(thread_id=thread_id).first()
    
    if existing_decision_record:
        existing_decision_record.request_type = identified_request_type
        existing_decision_record.extracted_fields = provided_fields
        existing_decision_record.missing_fields = missing_fields
        existing_decision_record.mandatory_fields = mandatory_fields
        existing_decision_record.outcome = final_decision_outcome
        existing_decision_record.rationale = decision_rationale
        existing_decision_record.risk_score = calculated_risk_score
        existing_decision_record.confidence_score = calculated_confidence_score
    else:
        new_decision_record = Decision(
            thread_id=thread_id,
            request_type=identified_request_type,
            extracted_fields=provided_fields,
            missing_fields=missing_fields,
            mandatory_fields=mandatory_fields,
            outcome=final_decision_outcome,
            rationale=decision_rationale,
            risk_score=calculated_risk_score,
            confidence_score=calculated_confidence_score
        )
        database_session.add(new_decision_record)

def _create_audit_log(database_session, thread_id: str, message_text: str, 
                      identified_request_type: str, missing_fields, final_decision_outcome):
    audit_log_entry = AuditLog(
        thread_id=thread_id,
        action="process_message",
        input_data={"text": message_text},
        output_data={
            "request_type": identified_request_type,
            "missing_fields": missing_fields,
            "decision": final_decision_outcome
        }
    )
    database_session.add(audit_log_entry)

@app.post("/threads/{thread_id}/messages")
def process_incoming_message(thread_id: str, message: MessageInput):
    with get_db_session() as database_session:
        _save_user_message(database_session, thread_id, message.text)
        
        complete_conversation_text = _build_conversation_history(database_session, thread_id)
        
        similar_historical_tickets = find_similar_historical_tickets(complete_conversation_text, number_of_tickets_to_retrieve=5)
        
        identified_request_type, provided_fields, missing_fields, mandatory_fields = _analyze_security_request(complete_conversation_text, similar_historical_tickets)
        
        next_question_to_ask, final_decision_outcome, decision_rationale, calculated_risk_score, calculated_confidence_score = _determine_next_action(
            complete_conversation_text, identified_request_type, provided_fields, missing_fields, similar_historical_tickets
        )
        
        _upsert_decision_record(
            database_session, thread_id, identified_request_type, 
            provided_fields, missing_fields, mandatory_fields,
            final_decision_outcome, decision_rationale, calculated_risk_score, calculated_confidence_score
        )
        
        _create_audit_log(database_session, thread_id, message.text, identified_request_type, missing_fields, final_decision_outcome)
    
    return {
        "request_type": identified_request_type,
        "risk_score": calculated_risk_score,
        "confidence_score": calculated_confidence_score,
        "missing_fields": missing_fields,
        "next_question": next_question_to_ask,
        "final_decision": final_decision_outcome,
        "rationale": decision_rationale
    }

@app.get("/health")
def comprehensive_risk_posture():
    
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    with get_db_session() as database_session:
        # Top 5 riskiest recent decisions
        riskiest_decisions = database_session.query(Decision).filter(
            Decision.created_at >= thirty_days_ago,
            Decision.risk_score.isnot(None)
        ).order_by(Decision.risk_score.desc()).limit(5).all()
        
        total_requests = database_session.query(Decision).filter(
            Decision.created_at >= thirty_days_ago
        ).count()
        
        risky_requests = []
        for d in riskiest_decisions:
            messages = database_session.query(Message).filter_by(
                thread_id=d.thread_id,
                role="user"
            ).order_by(Message.timestamp).all()
            
            request_text = " ".join([m.text for m in messages])
            
            risky_requests.append({
                "request_type": d.request_type,
                "risk_score": d.risk_score,
                "outcome": d.outcome,
                "request_summary": request_text[:200],
                "extracted_fields": d.extracted_fields,
                "rationale": d.rationale
            })
        
        formatted_requests = []
        for i, request in enumerate(risky_requests, start=1):
            formatted_request = (
                f"{i}. [{request['request_type']}] Risk: {request['risk_score']}/100 - {request['outcome']}\n"
                f"Summary: {request['request_summary']}\n"
                f"Rationale: {request['rationale']}"
            )
            formatted_requests.append(formatted_request)
        
        risky_requests_text = "\n".join(formatted_requests)
        
        llm_prompt = f"""Analyze these 5 riskiest security requests from the last 30 days. Identify patterns and concerns.

                        Riskiest Requests:
                        {risky_requests_text}

                        Provide JSON with:
                        {{
                            "patterns_detected": ["pattern 1", "pattern 2"],
                            "common_risk_factors": ["factor 1", "factor 2"],
                            "recommendations": ["action 1", "action 2"],
                            "alert_level": "low/medium/high"
                        }}"""

        try:
            pattern_analysis = _call_llm_for_json(llm_prompt)
        except Exception:
            pattern_analysis = {"error": "Analysis unavailable"}
    
    return {
        "period": "last_30_days",
        "total_requests": total_requests,
        "top_5_riskiest": risky_requests,
        "pattern_analysis": pattern_analysis
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
