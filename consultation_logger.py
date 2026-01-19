"""
Consultation Logging and Notification System
"""

import json
import os
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('consultation_logs.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ConsultationLog:
    """Data class for consultation logs"""
    id: str
    action: str  # scheduled, updated, confirmed, cancelled
    consultation_id: str
    user_name: str
    user_email: str
    user_phone: str
    company: str
    preferred_date: str
    preferred_time: str
    message: str
    status: str
    timestamp: str
    ip_address: str = ""
    user_agent: str = ""

@dataclass
class TeamMember:
    """Data class for team members who should be notified"""
    name: str
    email: str
    role: str
    phone: str = ""

class ConsultationLogger:
    """Handles logging and notifications for consultation requests"""
    
    def __init__(self, log_file: str = "consultation_logs.json", team_file: str = "team_members.json"):
        self.log_file = log_file
        self.team_file = team_file
        self.logs = self._load_logs()
        self.team_members = self._load_team_members()
        
        # Email configuration (you can set these in environment variables)
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "ask@softtechniques.com")
        
        # Initialize default team members if none exist
        if not self.team_members:
            self._initialize_default_team()
    
    def _load_logs(self) -> List[Dict]:
        """Load existing consultation logs from file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_logs(self):
        """Save consultation logs to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def _load_team_members(self) -> List[Dict]:
        """Load team members from file"""
        if os.path.exists(self.team_file):
            try:
                with open(self.team_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_team_members(self):
        """Save team members to file"""
        with open(self.team_file, 'w') as f:
            json.dump(self.team_members, f, indent=2)
    
    def _initialize_default_team(self):
        """Initialize default team members"""
        default_team = [
            {
                "name": "Sales Team",
                "email": "ask@softtechniques.com",
                "role": "Sales Representative",
                "phone": "(888) 324-6560"
            },
            {
                "name": "Project Manager",
                "email": "ask@softtechniques.com",
                "role": "Project Manager",
                "phone": "(888) 324-6560"
            }
        ]
        self.team_members = default_team
        self._save_team_members()
    
    def log_consultation_action(self, 
                               action: str,
                               consultation_id: str,
                               user_name: str,
                               user_email: str,
                               user_phone: str = "",
                               company: str = "",
                               preferred_date: str = "",
                               preferred_time: str = "",
                               message: str = "",
                               status: str = "pending",
                               ip_address: str = "",
                               user_agent: str = "") -> Dict:
        """Log a consultation action"""
        
        log_entry = ConsultationLog(
            id=f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{consultation_id}",
            action=action,
            consultation_id=consultation_id,
            user_name=user_name,
            user_email=user_email,
            user_phone=user_phone,
            company=company,
            preferred_date=preferred_date,
            preferred_time=preferred_time,
            message=message,
            status=status,
            timestamp=datetime.now().isoformat(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Add to logs
        self.logs.append(asdict(log_entry))
        self._save_logs()
        
        # Log to file
        logging.info(f"Consultation {action}: {consultation_id} - {user_name} ({user_email})")
        
        # Send notifications
        self._send_notifications(log_entry)
        
        return {
            "success": True,
            "log_id": log_entry.id,
            "message": f"Consultation {action} logged successfully"
        }
    
    def _send_notifications(self, log_entry: ConsultationLog):
        """Send notifications to team members"""
        try:
            if log_entry.action == "scheduled":
                self._send_new_consultation_notification(log_entry)
            elif log_entry.action in ["confirmed", "cancelled"]:
                self._send_status_update_notification(log_entry)
        except Exception as e:
            logging.error(f"Error sending notifications: {e}")
    
    def _send_new_consultation_notification(self, log_entry: ConsultationLog):
        """Send notification for new consultation request"""
        subject = f"New Consultation Request - {log_entry.user_name}"
        
        body = f"""
New consultation request received:

Consultation ID: {log_entry.consultation_id}
Name: {log_entry.user_name}
Email: {log_entry.user_email}
Phone: {log_entry.user_phone}
Company: {log_entry.company}
Preferred Date: {log_entry.preferred_date}
Preferred Time: {log_entry.preferred_time}
Message: {log_entry.message}
Status: {log_entry.status}
Timestamp: {log_entry.timestamp}

Please contact this prospect within 24 hours to confirm their appointment.

Contact Information:
- Phone: +1 (555) 012-3456
- Email: ask@softtechniques.com

This is an automated notification from your consultation scheduling system.
        """
        
        self._send_email_to_team(subject, body)
    
    def _send_status_update_notification(self, log_entry: ConsultationLog):
        """Send notification for status updates"""
        subject = f"Consultation Status Update - {log_entry.consultation_id}"
        
        body = f"""
Consultation status updated:

Consultation ID: {log_entry.consultation_id}
Name: {log_entry.user_name}
Email: {log_entry.user_email}
New Status: {log_entry.status}
Action: {log_entry.action}
Timestamp: {log_entry.timestamp}

Please follow up as needed.

This is an automated notification from your consultation scheduling system.
        """
        
        self._send_email_to_team(subject, body)
    
    def _send_email_to_team(self, subject: str, body: str):
        """Send email to all team members"""
        if not self.smtp_username or not self.smtp_password:
            logging.warning("Email credentials not configured. Skipping email notification.")
            return
        
        try:
            for member in self.team_members:
                self._send_email(member["email"], subject, body)
        except Exception as e:
            logging.error(f"Error sending email to team: {e}")
    
    def _send_email(self, to_email: str, subject: str, body: str):
        """Send individual email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            text = msg.as_string()
            server.sendmail(self.from_email, to_email, text)
            server.quit()
            
            logging.info(f"Email sent to {to_email}")
        except Exception as e:
            logging.error(f"Error sending email to {to_email}: {e}")
    
    def get_recent_logs(self, hours: int = 24) -> List[Dict]:
        """Get recent consultation logs"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_logs = []
        
        for log in self.logs:
            log_time = datetime.fromisoformat(log["timestamp"])
            if log_time >= cutoff_time:
                recent_logs.append(log)
        
        return sorted(recent_logs, key=lambda x: x["timestamp"], reverse=True)
    
    def get_logs_by_status(self, status: str) -> List[Dict]:
        """Get logs by status"""
        return [log for log in self.logs if log["status"] == status]
    
    def get_logs_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get logs by date range"""
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        filtered_logs = []
        for log in self.logs:
            log_time = datetime.fromisoformat(log["timestamp"])
            if start <= log_time <= end:
                filtered_logs.append(log)
        
        return sorted(filtered_logs, key=lambda x: x["timestamp"], reverse=True)
    
    def add_team_member(self, name: str, email: str, role: str, phone: str = "") -> Dict:
        """Add a new team member"""
        team_member = {
            "name": name,
            "email": email,
            "role": role,
            "phone": phone
        }
        
        self.team_members.append(team_member)
        self._save_team_members()
        
        return {
            "success": True,
            "message": f"Team member {name} added successfully"
        }
    
    def remove_team_member(self, email: str) -> Dict:
        """Remove a team member"""
        original_count = len(self.team_members)
        self.team_members = [member for member in self.team_members if member["email"] != email]
        
        if len(self.team_members) < original_count:
            self._save_team_members()
            return {"success": True, "message": f"Team member {email} removed successfully"}
        else:
            return {"success": False, "message": "Team member not found"}
    
    def get_team_members(self) -> List[Dict]:
        """Get all team members"""
        return self.team_members
    
    def get_consultation_stats(self) -> Dict:
        """Get consultation statistics from actual consultation requests"""
        # Import here to avoid circular imports
        from scheduling_system import consultation_scheduler
        
        # Get actual consultation requests, not logs
        requests = consultation_scheduler.get_all_requests()
        
        total_requests = len(requests)
        pending_requests = len([req for req in requests if req["status"] == "pending"])
        confirmed_requests = len([req for req in requests if req["status"] == "confirmed"])
        completed_requests = len([req for req in requests if req["status"] == "completed"])
        cancelled_requests = len([req for req in requests if req["status"] == "cancelled"])
        
        # Recent activity (last 7 days) - count requests created in last 7 days
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(days=7)
        recent_requests = 0
        
        for req in requests:
            try:
                created_at = datetime.fromisoformat(req["created_at"])
                if created_at >= cutoff_time:
                    recent_requests += 1
            except:
                # If date parsing fails, skip this request
                continue
        
        return {
            "total_requests": total_requests,
            "pending_requests": pending_requests,
            "confirmed_requests": confirmed_requests,
            "completed_requests": completed_requests,
            "cancelled_requests": cancelled_requests,
            "recent_requests_7_days": recent_requests,
            "team_members_count": len(self.team_members)
        }
    
    def clear_all_logs(self) -> Dict:
        """Clear all consultation logs"""
        original_count = len(self.logs)
        self.logs = []
        self._save_logs()
        
        logging.info(f"All consultation logs cleared. {original_count} logs removed.")
        
        return {
            "success": True,
            "message": f"All consultation logs cleared successfully. {original_count} logs removed.",
            "logs_removed": original_count
        }

# Initialize the logger
consultation_logger = ConsultationLogger()
