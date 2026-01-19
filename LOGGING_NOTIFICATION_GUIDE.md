# Consultation Logging & Notification System Guide

## 📊 **Complete Logging & Notification System Implemented!**

I've implemented a comprehensive logging and notification system that automatically logs all consultation activities and notifies your team members when new consultations are scheduled.

## 🚀 **New Capabilities:**

### ✅ **Automatic Logging:**
- **Every consultation action is logged** with timestamp, user details, and status
- **IP address and user agent tracking** for security and analytics
- **Status change tracking** (pending → confirmed → completed)
- **Persistent storage** in JSON files for data retention
- **File-based logging** for system monitoring

### ✅ **Team Notifications:**
- **Automatic email notifications** to team members when consultations are scheduled
- **Status update notifications** when consultations are confirmed/cancelled
- **Team member management** - add/remove team members who receive notifications
- **Configurable email settings** for different SMTP providers

### ✅ **Admin Dashboard:**
- **Real-time statistics** - total requests, pending, confirmed, completed
- **Recent activity logs** with filtering by time range and status
- **Team member management** interface
- **Consultation status updates** directly from the dashboard
- **Export capabilities** for reporting and analysis

## 🔧 **How It Works:**

### **1. Automatic Logging Process:**
```
User schedules consultation → System logs action → Team gets notified → Admin can view in dashboard
```

### **2. Logging Details:**
- **Consultation ID** - Unique identifier for tracking
- **User Information** - Name, email, phone, company
- **Scheduling Details** - Preferred date, time, message
- **Technical Data** - IP address, user agent, timestamp
- **Status Tracking** - Current status and status changes

### **3. Notification System:**
- **New Consultation** - Email sent to all team members
- **Status Updates** - Notifications for confirmed/cancelled appointments
- **Team Management** - Add/remove team members who receive notifications

## 🌐 **Access Points:**

### **Admin Dashboard:**
```
http://localhost:8000/admin-dashboard
```

### **API Endpoints:**
- `GET /admin/stats` - Get consultation statistics
- `GET /admin/logs/recent` - Get recent consultation logs
- `GET /admin/logs/status/{status}` - Get logs by status
- `GET /admin/logs/date-range` - Get logs by date range
- `GET /admin/team` - Get all team members
- `POST /admin/team/add` - Add new team member
- `DELETE /admin/team/remove/{email}` - Remove team member

## 📧 **Email Configuration:**

### **Environment Variables:**
Add these to your `.env` file for email notifications:
```env
# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=ask@akenotech.com
```

### **Gmail Setup:**
1. Enable 2-factor authentication on your Gmail account
2. Generate an "App Password" for the chatbot
3. Use the app password in `SMTP_PASSWORD`
4. Set `SMTP_USERNAME` to your Gmail address

### **Other Email Providers:**
- **Outlook:** `smtp-mail.outlook.com:587`
- **Yahoo:** `smtp.mail.yahoo.com:587`
- **Custom SMTP:** Configure your own SMTP server

## 🧪 **Test the System:**

### **Start your chatbot:**
```bash
python run.py
```

### **Test logging through chat:**
1. Go to: `http://localhost:8000/chat-interface`
2. Ask: "I want to schedule a consultation"
3. Fill out the scheduling form
4. Check the admin dashboard for the logged activity

### **Test admin dashboard:**
1. Go to: `http://localhost:8000/admin-dashboard`
2. View statistics and recent logs
3. Add team members
4. Update consultation statuses

### **Test API endpoints:**
```bash
# Get statistics
curl "http://localhost:8000/admin/stats"

# Get recent logs
curl "http://localhost:8000/admin/logs/recent?hours=24"

# Get logs by status
curl "http://localhost:8000/admin/logs/status/pending"

# Add team member
curl -X POST "http://localhost:8000/admin/team/add?name=John%20Doe&email=john@example.com&role=Sales%20Rep&phone=555-1234"

# Get team members
curl "http://localhost:8000/admin/team"
```

## 📊 **Admin Dashboard Features:**

### **Statistics Overview:**
- **Total Requests** - All consultation requests ever received
- **Pending Requests** - Awaiting confirmation
- **Confirmed Requests** - Confirmed appointments
- **Recent Requests** - Last 7 days activity

### **Recent Logs:**
- **Time Range Filtering** - 24 hours, 7 days, 30 days
- **Status Filtering** - Filter by pending, confirmed, completed, cancelled
- **Detailed Information** - User details, timestamps, actions
- **Real-time Updates** - Refresh to see latest activity

### **Team Management:**
- **Add Team Members** - Name, email, role, phone
- **Remove Team Members** - Remove from notification list
- **View Team List** - See all team members and their roles

### **Consultation Management:**
- **View All Requests** - Complete list of consultation requests
- **Update Status** - Change status directly from dashboard
- **Status Tracking** - See status changes over time

## 📋 **Data Storage:**

### **Log Files:**
- **`consultation_logs.json`** - All consultation activities and logs
- **`consultation_requests.json`** - All consultation requests
- **`team_members.json`** - Team members who receive notifications
- **`consultation_logs.log`** - System log file for debugging

### **Log Structure:**
```json
{
  "id": "log_20241201_143022_ABC12345",
  "action": "scheduled",
  "consultation_id": "ABC12345",
  "user_name": "John Doe",
  "user_email": "john@example.com",
  "user_phone": "555-1234",
  "company": "Acme Corp",
  "preferred_date": "2024-12-15",
  "preferred_time": "2:00 PM",
  "message": "Interested in agentic AI automation",
  "status": "pending",
  "timestamp": "2024-12-01T14:30:22.123456",
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0..."
}
```

## 🔔 **Notification Examples:**

### **New Consultation Notification:**
```
Subject: New Consultation Request - John Doe

New consultation request received:

Consultation ID: ABC12345
Name: John Doe
Email: john@example.com
Phone: 555-1234
Company: Acme Corp
Preferred Date: 2024-12-15
Preferred Time: 2:00 PM
Message: Interested in agentic AI automation
Status: pending
Timestamp: 2024-12-01T14:30:22

Please contact this prospect within 24 hours to confirm their appointment.

Contact Information:
- Phone: (888) 324-6560
- Email: ask@akenotech.com

This is an automated notification from your consultation scheduling system.
```

### **Status Update Notification:**
```
Subject: Consultation Status Update - ABC12345

Consultation status updated:

Consultation ID: ABC12345
Name: John Doe
Email: john@example.com
New Status: confirmed
Action: updated
Timestamp: 2024-12-01T15:45:30

Please follow up as needed.

This is an automated notification from your consultation scheduling system.
```

## 🚀 **Ready to Use:**

### **Start your chatbot:**
```bash
python run.py
```

### **Access the admin dashboard:**
- **Admin Dashboard:** `http://localhost:8000/admin-dashboard`
- **Chat Interface:** `http://localhost:8000/chat-interface`
- **Direct Scheduling:** `http://localhost:8000/schedule-consultation`

### **Configure email notifications:**
1. Add email settings to `.env` file
2. Add team members through admin dashboard
3. Test by scheduling a consultation

## 📈 **Benefits:**

### ✅ **For Your Team:**
- **Immediate notifications** when consultations are scheduled
- **Complete activity tracking** for all consultation requests
- **Easy status management** through admin dashboard
- **Team member management** for notification distribution

### ✅ **For Your Business:**
- **Never miss a lead** - automatic logging and notifications
- **Complete audit trail** of all consultation activities
- **Data-driven insights** through statistics and reporting
- **Professional follow-up** with 24-hour response commitment

### ✅ **For Security & Compliance:**
- **IP address tracking** for security monitoring
- **Complete audit logs** for compliance requirements
- **Data persistence** for long-term record keeping
- **User agent tracking** for analytics and security

**Your consultation system now has complete logging and notification capabilities! Every consultation is automatically logged, your team gets notified immediately, and you have a full admin dashboard to manage everything.** 🎉

---

## 📝 **Quick Start:**

1. **Start your chatbot:** `python run.py`
2. **Access admin dashboard:** Visit `/admin-dashboard`
3. **Add team members:** Use the team management interface
4. **Configure email:** Add SMTP settings to `.env` file
5. **Test the system:** Schedule a consultation and check notifications

Your chatbot is now a complete lead generation, scheduling, and management system with full logging and notification capabilities!
