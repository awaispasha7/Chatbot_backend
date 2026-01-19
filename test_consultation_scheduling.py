#!/usr/bin/env python3
"""
Test script for consultation scheduling through chat
"""

import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_consultation_scheduling_through_chat():
    """Test consultation scheduling through the chat interface"""
    print("🧪 Testing Consultation Scheduling Through Chat...")
    
    # Test consultation message
    consultation_message = """
    Hi, I want to schedule a consultation. My name is Shehryar, my email is shahzadashehryar16@gmail.com, 
    my company is Softec Techniques, my phone is 555 2348769, and I'd like to schedule for September 29, 2025 at 7 PM. 
    I'm interested in learning about agentic AI for our software development processes.
    """
    
    try:
        # Send chat message
        print("📤 Sending consultation request through chat...")
        response = requests.post(f"{BASE_URL}/chat", json={
            "message": consultation_message,
            "session_id": "test_consultation_session"
        })
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat response received!")
            print(f"   Response: {data['response'][:200]}...")
            
            # Check if consultation was actually scheduled
            print("\n🔍 Checking if consultation was scheduled...")
            consultations_response = requests.get(f"{BASE_URL}/consultation/all")
            
            if consultations_response.status_code == 200:
                consultations_data = consultations_response.json()
                consultations = consultations_data.get('requests', [])
                
                # Look for our test consultation
                test_consultation = None
                for consultation in consultations:
                    if consultation.get('email') == 'shahzadashehryar16@gmail.com':
                        test_consultation = consultation
                        break
                
                if test_consultation:
                    print("✅ Consultation was successfully scheduled!")
                    print(f"   Consultation ID: {test_consultation['id']}")
                    print(f"   Name: {test_consultation['name']}")
                    print(f"   Email: {test_consultation['email']}")
                    print(f"   Company: {test_consultation['company']}")
                    print(f"   Phone: {test_consultation['phone']}")
                    print(f"   Status: {test_consultation['status']}")
                    return test_consultation['id']
                else:
                    print("❌ Consultation was not found in the system")
                    print("   This means the chatbot responded but didn't actually schedule the consultation")
            else:
                print(f"❌ Error checking consultations: {consultations_response.status_code}")
        else:
            print(f"❌ Error sending chat message: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception during test: {e}")
    
    return None

def test_admin_dashboard_data():
    """Test if admin dashboard shows the consultation"""
    print("\n📊 Testing Admin Dashboard Data...")
    
    try:
        # Get statistics
        stats_response = requests.get(f"{BASE_URL}/admin/stats")
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            stats = stats_data.get('stats', {})
            print("✅ Admin statistics retrieved!")
            print(f"   Total Requests: {stats.get('total_requests', 0)}")
            print(f"   Pending Requests: {stats.get('pending_requests', 0)}")
            print(f"   Recent Requests (7 days): {stats.get('recent_requests_7_days', 0)}")
        else:
            print(f"❌ Error getting stats: {stats_response.status_code}")
        
        # Get recent logs
        logs_response = requests.get(f"{BASE_URL}/admin/logs/recent?hours=1")
        if logs_response.status_code == 200:
            logs_data = logs_response.json()
            logs = logs_data.get('logs', [])
            print(f"✅ Recent logs retrieved! Found {len(logs)} logs")
            
            # Look for consultation logs
            consultation_logs = [log for log in logs if log.get('action') == 'scheduled']
            if consultation_logs:
                print(f"   Found {len(consultation_logs)} consultation scheduling logs")
                for log in consultation_logs:
                    print(f"     - {log.get('user_name')} ({log.get('user_email')}) - {log.get('consultation_id')}")
            else:
                print("   No consultation scheduling logs found")
        else:
            print(f"❌ Error getting logs: {logs_response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception getting admin data: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Consultation Scheduling Tests")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server is not running. Please start the chatbot first:")
            print("   python run.py")
            return
    except:
        print("❌ Cannot connect to server. Please start the chatbot first:")
        print("   python run.py")
        return
    
    print("✅ Server is running!")
    
    # Test consultation scheduling
    consultation_id = test_consultation_scheduling_through_chat()
    
    # Test admin dashboard data
    test_admin_dashboard_data()
    
    print("\n" + "=" * 50)
    print("🎉 Tests completed!")
    
    if consultation_id:
        print(f"\n✅ Consultation {consultation_id} was successfully scheduled!")
        print("📊 Check the admin dashboard at:")
        print(f"   {BASE_URL}/admin-dashboard")
    else:
        print("\n❌ Consultation scheduling test failed!")
        print("🔧 The chatbot may need to be updated to properly handle consultation requests.")

if __name__ == "__main__":
    main()
