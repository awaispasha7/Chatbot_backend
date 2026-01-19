"""
Consultation Scheduling System for the Chatbot
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import uuid
from consultation_logger import consultation_logger

@dataclass
class ConsultationRequest:
    """Data class for consultation requests"""
    id: str
    name: str
    email: str
    phone: str
    company: str
    preferred_date: str
    preferred_time: str
    timezone: str
    message: str
    status: str  # pending, confirmed, completed, cancelled
    created_at: str
    confirmed_at: Optional[str] = None

class ConsultationScheduler:
    """Handles consultation scheduling functionality"""
    
    def __init__(self, data_file: str = "consultation_requests.json"):
        self.data_file = data_file
        self.requests = self._load_requests()
        
        # Available time slots (you can customize these)
        self.available_slots = [
            "9:00 AM", "10:00 AM", "11:00 AM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM"
        ]
        
        # Available days (next 14 days)
        self.available_days = self._generate_available_days()
    
    def _load_requests(self) -> List[Dict]:
        """Load existing consultation requests from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_requests(self):
        """Save consultation requests to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.requests, f, indent=2)
    
    def _generate_available_days(self) -> List[str]:
        """Generate available days for the next 14 days"""
        days = []
        today = datetime.now()
        
        for i in range(1, 15):  # Next 14 days
            date = today + timedelta(days=i)
            # Skip weekends (optional - you can remove this)
            if date.weekday() < 5:  # Monday = 0, Friday = 4
                days.append(date.strftime("%Y-%m-%d"))
        
        return days
    
    def get_available_slots(self) -> Dict[str, List[str]]:
        """Get available consultation slots (excluding booked times)"""
        # Get all booked time slots
        booked_slots = self._get_booked_slots()
        
        # Filter out booked slots for each day
        available_slots_by_day = {}
        for day in self.available_days:
            day_booked_times = booked_slots.get(day, [])
            available_times = [time for time in self.available_slots if time not in day_booked_times]
            if available_times:  # Only include days with available slots
                available_slots_by_day[day] = available_times
        
        return {
            "available_days": list(available_slots_by_day.keys()),
            "available_times": self.available_slots,  # All possible times
            "available_slots_by_day": available_slots_by_day,  # Day-specific available times
            "timezone": "EST"  # You can make this configurable
        }
    
    def _get_booked_slots(self) -> Dict[str, List[str]]:
        """Get all booked time slots organized by date"""
        booked_slots = {}
        
        for request in self.requests:
            # Only count confirmed and pending requests as "booked"
            if request["status"] in ["confirmed", "pending"] and request["preferred_date"] and request["preferred_time"]:
                date = request["preferred_date"]
                time = request["preferred_time"]
                
                if date not in booked_slots:
                    booked_slots[date] = []
                if time not in booked_slots[date]:
                    booked_slots[date].append(time)
        
        return booked_slots
    
    def is_time_slot_available(self, date: str, time: str) -> bool:
        """Check if a specific time slot is available"""
        booked_slots = self._get_booked_slots()
        day_booked_times = booked_slots.get(date, [])
        return time not in day_booked_times
    
    def schedule_consultation(self, 
                            name: str, 
                            email: str, 
                            phone: str = "",
                            company: str = "",
                            preferred_date: str = "",
                            preferred_time: str = "",
                            message: str = "",
                            ip_address: str = "",
                            user_agent: str = "") -> Dict:
        """Schedule a new consultation"""
        
        # Check if the time slot is available
        if preferred_date and preferred_time:
            if not self.is_time_slot_available(preferred_date, preferred_time):
                return {
                    "success": False,
                    "message": f"Sorry, the time slot {preferred_time} on {preferred_date} is no longer available. Please select a different time.",
                    "suggestion": "Try refreshing the page to see updated available times."
                }
        
        # Generate unique ID
        consultation_id = str(uuid.uuid4())[:8]
        
        # Create consultation request
        request = ConsultationRequest(
            id=consultation_id,
            name=name,
            email=email,
            phone=phone,
            company=company,
            preferred_date=preferred_date,
            preferred_time=preferred_time,
            timezone="EST",
            message=message,
            status="pending",
            created_at=datetime.now().isoformat()
        )
        
        # Add to requests
        self.requests.append(asdict(request))
        self._save_requests()
        
        # Log the consultation action
        log_result = consultation_logger.log_consultation_action(
            action="scheduled",
            consultation_id=consultation_id,
            user_name=name,
            user_email=email,
            user_phone=phone,
            company=company,
            preferred_date=preferred_date,
            preferred_time=preferred_time,
            message=message,
            status="pending",
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return {
            "success": True,
            "consultation_id": consultation_id,
            "message": f"Consultation request created successfully! Your consultation ID is {consultation_id}.",
            "next_steps": "Our team will contact you within 24 hours to confirm your appointment.",
            "contact_info": {
                "phone": "+1 (555) 012-3456",
                "email": "ask@softtechniques.com"
            },
            "logged": log_result["success"]
        }
    
    def get_consultation_status(self, consultation_id: str) -> Dict:
        """Get status of a consultation request"""
        for request in self.requests:
            if request["id"] == consultation_id:
                return {
                    "found": True,
                    "status": request["status"],
                    "created_at": request["created_at"],
                    "confirmed_at": request.get("confirmed_at"),
                    "details": request
                }
        
        return {"found": False, "message": "Consultation request not found"}
    
    def get_all_requests(self) -> List[Dict]:
        """Get all consultation requests (for admin use)"""
        return self.requests
    
    def update_consultation_status(self, consultation_id: str, status: str) -> Dict:
        """Update consultation status (for admin use)"""
        for request in self.requests:
            if request["id"] == consultation_id:
                old_status = request["status"]
                request["status"] = status
                if status == "confirmed":
                    request["confirmed_at"] = datetime.now().isoformat()
                self._save_requests()
                
                # Log the status update
                log_result = consultation_logger.log_consultation_action(
                    action="updated",
                    consultation_id=consultation_id,
                    user_name=request["name"],
                    user_email=request["email"],
                    user_phone=request["phone"],
                    company=request["company"],
                    preferred_date=request["preferred_date"],
                    preferred_time=request["preferred_time"],
                    message=request["message"],
                    status=status
                )
                
                return {
                    "success": True, 
                    "message": f"Consultation {consultation_id} status updated from {old_status} to {status}",
                    "logged": log_result["success"]
                }
        
        return {"success": False, "message": "Consultation request not found"}
    
    def delete_consultation(self, consultation_id: str) -> Dict:
        """Delete a consultation request (for admin use)"""
        original_count = len(self.requests)
        self.requests = [request for request in self.requests if request["id"] != consultation_id]
        
        if len(self.requests) < original_count:
            self._save_requests()
            
            # Log the deletion
            log_result = consultation_logger.log_consultation_action(
                action="deleted",
                consultation_id=consultation_id,
                user_name="Admin",
                user_email="admin@system",
                status="deleted"
            )
            
            return {
                "success": True,
                "message": f"Consultation {consultation_id} deleted successfully",
                "logged": log_result["success"]
            }
        else:
            return {"success": False, "message": "Consultation request not found"}

# Initialize the scheduler
consultation_scheduler = ConsultationScheduler()
