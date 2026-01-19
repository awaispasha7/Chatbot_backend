#!/usr/bin/env python3
"""
Test script for the consultation logging and notification system
"""

import requests
import json
import time
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_consultation_scheduling():
    """Test consultation scheduling and logging"""
    print("🧪 Testing Consultation Scheduling and Logging...")
    
    # Test data
    test_consultation = {
        "name": "Test User",
        "email": "test@example.com",
        "phone": "555-1234",
        "company": "Test Company",
        "preferred_date": "2024-12-15",
        "preferred_time": "2:00 PM",
        "message": "Testing the logging system"
    }
    
    try:
        # Schedule consultation
        print("📅 Scheduling consultation...")
        response = requests.post(f"{BASE_URL}/consultation/schedule", json=test_consultation)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Consultation scheduled successfully!")
            print(f"   Consultation ID: {data['consultation_id']}")
            print(f"   Logged: {data.get('logged', 'Unknown')}")
            return data['consultation_id']
        else:
            print(f"❌ Error scheduling consultation: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception during scheduling: {e}")
        return None

def test_admin_stats():
    """Test admin statistics endpoint"""
    print("\n📊 Testing Admin Statistics...")
    
    try:
        response = requests.get(f"{BASE_URL}/admin/stats")
        
        if response.status_code == 200:
            data = response.json()
            stats = data['stats']
            print("✅ Statistics retrieved successfully!")
            print(f"   Total Requests: {stats['total_requests']}")
            print(f"   Pending Requests: {stats['pending_requests']}")
            print(f"   Confirmed Requests: {stats['confirmed_requests']}")
            print(f"   Recent Requests (7 days): {stats['recent_requests_7_days']}")
            print(f"   Team Members: {stats['team_members_count']}")
        else:
            print(f"❌ Error getting stats: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception getting stats: {e}")

def test_recent_logs():
    """Test recent logs endpoint"""
    print("\n📋 Testing Recent Logs...")
    
    try:
        response = requests.get(f"{BASE_URL}/admin/logs/recent?hours=24")
        
        if response.status_code == 200:
            data = response.json()
            logs = data['logs']
            print(f"✅ Recent logs retrieved successfully!")
            print(f"   Total logs: {len(logs)}")
            
            if logs:
                latest_log = logs[0]
                print(f"   Latest log:")
                print(f"     Action: {latest_log['action']}")
                print(f"     Consultation ID: {latest_log['consultation_id']}")
                print(f"     User: {latest_log['user_name']} ({latest_log['user_email']})")
                print(f"     Status: {latest_log['status']}")
                print(f"     Timestamp: {latest_log['timestamp']}")
        else:
            print(f"❌ Error getting logs: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception getting logs: {e}")

def test_team_management():
    """Test team management endpoints"""
    print("\n👥 Testing Team Management...")
    
    try:
        # Get current team members
        response = requests.get(f"{BASE_URL}/admin/team")
        
        if response.status_code == 200:
            data = response.json()
            team_members = data['team_members']
            print(f"✅ Team members retrieved successfully!")
            print(f"   Current team members: {len(team_members)}")
            
            for member in team_members:
                print(f"     - {member['name']} ({member['email']}) - {member['role']}")
        else:
            print(f"❌ Error getting team members: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception getting team members: {e}")

def test_consultation_status_update(consultation_id):
    """Test consultation status update"""
    if not consultation_id:
        print("\n⏭️ Skipping status update test (no consultation ID)")
        return
        
    print(f"\n🔄 Testing Status Update for {consultation_id}...")
    
    try:
        # Update status to confirmed
        response = requests.put(f"{BASE_URL}/consultation/update-status/{consultation_id}?status=confirmed")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status updated successfully!")
            print(f"   Message: {data['message']}")
            print(f"   Logged: {data.get('logged', 'Unknown')}")
        else:
            print(f"❌ Error updating status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception updating status: {e}")

def test_consultation_requests():
    """Test consultation requests endpoint"""
    print("\n📅 Testing Consultation Requests...")
    
    try:
        response = requests.get(f"{BASE_URL}/consultation/all")
        
        if response.status_code == 200:
            data = response.json()
            requests = data['requests']
            print(f"✅ Consultation requests retrieved successfully!")
            print(f"   Total requests: {len(requests)}")
            
            if requests:
                latest_request = requests[-1]  # Most recent
                print(f"   Latest request:")
                print(f"     ID: {latest_request['id']}")
                print(f"     Name: {latest_request['name']}")
                print(f"     Email: {latest_request['email']}")
                print(f"     Status: {latest_request['status']}")
                print(f"     Created: {latest_request['created_at']}")
        else:
            print(f"❌ Error getting requests: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception getting requests: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Consultation Logging System Tests")
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
    
    # Run tests
    consultation_id = test_consultation_scheduling()
    test_admin_stats()
    test_recent_logs()
    test_team_management()
    test_consultation_status_update(consultation_id)
    test_consultation_requests()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\n📊 Check the admin dashboard at:")
    print(f"   {BASE_URL}/admin-dashboard")
    print("\n📋 Check the consultation logs at:")
    print(f"   {BASE_URL}/admin/logs/recent")

if __name__ == "__main__":
    main()




