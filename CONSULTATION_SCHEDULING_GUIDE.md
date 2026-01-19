# Consultation Scheduling System Guide

## 📅 **Automatic Consultation Scheduling Added!**

I've implemented a comprehensive consultation scheduling system that allows your chatbot to automatically schedule appointments for your company.

## 🚀 **New Capabilities:**

### ✅ **What the Chatbot Can Now Do:**
- **Automatic Scheduling** - Users can schedule consultations directly through the chat
- **Available Time Slots** - Shows available dates and times
- **Contact Collection** - Collects user details (name, email, phone, company)
- **Appointment Management** - Tracks and manages consultation requests
- **Status Updates** - Users can check their appointment status
- **Admin Dashboard** - View and manage all consultation requests

### ✅ **Scheduling Features:**
- **Available Slots** - Next 14 business days with time slots
- **Time Slots** - 9 AM, 10 AM, 11 AM, 1 PM, 2 PM, 3 PM, 4 PM
- **Contact Information** - Collects name, email, phone, company
- **Preferences** - Preferred date and time selection
- **Messages** - Optional project description
- **Confirmation** - Unique consultation ID for tracking

## 🔧 **How It Works:**

### **1. User Experience:**
1. User asks to schedule a consultation
2. Chatbot guides them to provide details
3. User fills out scheduling form
4. System creates consultation request
5. User receives confirmation with ID
6. Your team contacts them within 24 hours

### **2. Admin Experience:**
1. View all consultation requests
2. Update appointment status
3. Track consultation progress
4. Export contact information
5. Manage availability

## 🌐 **Access Points:**

### **Web Interface:**
```
http://localhost:8000/schedule-consultation
```

### **API Endpoints:**
- `GET /consultation/available-slots` - Get available time slots
- `POST /consultation/schedule` - Schedule new consultation
- `GET /consultation/status/{id}` - Check consultation status
- `GET /consultation/all` - View all consultations (admin)
- `PUT /consultation/update-status/{id}` - Update status (admin)

## 🧪 **Test the Scheduling System:**

### **Start your chatbot:**
```bash
python run.py
```

### **Test scheduling through chat:**
1. Go to: `http://localhost:8000/chat-interface`
2. Ask: "I want to schedule a consultation"
3. Follow the chatbot's guidance

### **Test direct scheduling:**
1. Go to: `http://localhost:8000/schedule-consultation`
2. Fill out the form
3. Submit the consultation request

### **Test API endpoints:**
```bash
# Get available slots
curl "http://localhost:8000/consultation/available-slots"

# Schedule consultation
curl -X POST "http://localhost:8000/consultation/schedule" \
     -H "Content-Type: application/json" \
     -d '{"name":"John Doe","email":"john@example.com","phone":"555-1234","company":"Acme Corp","preferred_date":"2024-01-15","preferred_time":"2:00 PM","message":"Interested in agentic AI for automation"}'

# Check status
curl "http://localhost:8000/consultation/status/{consultation_id}"

# View all consultations (admin)
curl "http://localhost:8000/consultation/all"
```

## 📊 **Scheduling Workflow:**

### **1. User Initiates Scheduling:**
```
User: "I want to schedule a consultation"
Chatbot: "Great! I can help you schedule a consultation. You can either:
- Fill out our scheduling form at /schedule-consultation
- Or provide your details here and I'll help you schedule

What's your name and email address?"
```

### **2. Information Collection:**
```
Chatbot: "Perfect! To schedule your consultation, I need:
- Your name: [collected]
- Email: [collected]
- Phone (optional): [collected]
- Company (optional): [collected]
- Preferred date and time
- Any specific message about your project

I'll create your consultation request and our team will contact you within 24 hours to confirm."
```

### **3. Confirmation:**
```
Chatbot: "Your consultation request has been created successfully! 
Your consultation ID is ABC12345. Our team will contact you within 24 hours to confirm your appointment.

You can also call us directly at (888) 324-6560 or email ask@akenotech.com for immediate assistance."
```

## 🎯 **Chatbot Integration:**

### **Updated System Prompt:**
The chatbot now includes scheduling guidance:
- Guides users to provide scheduling details
- Explains the scheduling process
- Offers both chat and direct contact options
- Mentions 24-hour confirmation timeline

### **Scheduling Triggers:**
The chatbot will offer scheduling when users:
- Ask to schedule a consultation
- Express interest in services
- Ask about getting started
- Request a demo or meeting
- Show readiness to begin a project

## 📋 **Data Storage:**

### **Consultation Data:**
- **File:** `consultation_requests.json`
- **Fields:** ID, name, email, phone, company, preferred date/time, message, status, timestamps
- **Statuses:** pending, confirmed, completed, cancelled

### **Available Slots:**
- **Days:** Next 14 business days (Monday-Friday)
- **Times:** 9 AM, 10 AM, 11 AM, 1 PM, 2 PM, 3 PM, 4 PM
- **Timezone:** EST (configurable)

## 🔧 **Customization Options:**

### **Available Time Slots:**
Edit `scheduling_system.py` to modify:
- Available days (currently 14 business days)
- Time slots (currently 7 slots per day)
- Weekend availability (currently weekdays only)
- Timezone (currently EST)

### **Form Fields:**
Customize the scheduling form in `static/consultation_scheduler.html`:
- Add/remove fields
- Change validation rules
- Modify styling
- Add additional options

## 🚀 **Ready to Use:**

### **Start your chatbot:**
```bash
python run.py
```

### **Access the scheduling system:**
- **Chat Interface:** `http://localhost:8000/chat-interface`
- **Direct Scheduling:** `http://localhost:8000/schedule-consultation`
- **API Documentation:** `http://localhost:8000/docs`

### **Admin Functions:**
- **View All Requests:** `GET /consultation/all`
- **Update Status:** `PUT /consultation/update-status/{id}`
- **Check Status:** `GET /consultation/status/{id}`

## 📈 **Benefits:**

### ✅ **For Users:**
- Easy scheduling through chat
- No need to call or email
- Immediate confirmation
- Clear next steps

### ✅ **For Your Business:**
- Automated lead capture
- Organized consultation requests
- Reduced manual scheduling
- Better prospect management
- 24/7 scheduling availability

**Your chatbot now has full consultation scheduling capabilities! Users can schedule appointments directly through the chat, and you'll have a complete system to manage all consultation requests.** 🎉

---

## 📝 **Quick Start:**

1. **Start your chatbot:** `python run.py`
2. **Test scheduling:** Ask "I want to schedule a consultation"
3. **Access scheduler:** Visit `/schedule-consultation`
4. **Manage requests:** Use `/consultation/all` endpoint
5. **Update status:** Use `/consultation/update-status/{id}` endpoint

Your chatbot is now a complete lead generation and scheduling system!
