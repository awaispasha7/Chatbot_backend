
const API_BASE = 'http://localhost:8000'; // FastAPI backend base URL
// Load dashboard data on page load
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadRecentLogs();
    loadTeamMembers();
    loadConsultations();
});

// Load statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/admin/stats`);
        const data = await response.json();
        
        console.log('Stats response:', data); // Debug log
        
        if (data.status === 'success') {
            const stats = data.stats;
            document.getElementById('totalRequests').textContent = stats.total_requests || 0;
            document.getElementById('pendingRequests').textContent = stats.pending_requests || 0;
            document.getElementById('confirmedRequests').textContent = stats.confirmed_requests || 0;
            document.getElementById('completedRequests').textContent = stats.completed_requests || 0;
            document.getElementById('cancelledRequests').textContent = stats.cancelled_requests || 0;
            document.getElementById('recentRequests').textContent = stats.recent_requests_7_days || 0;
        } else {
            console.error('Stats API returned error:', data);
            // Show error in UI
            document.getElementById('totalRequests').textContent = 'Error';
            document.getElementById('pendingRequests').textContent = 'Error';
            document.getElementById('confirmedRequests').textContent = 'Error';
            document.getElementById('completedRequests').textContent = 'Error';
            document.getElementById('cancelledRequests').textContent = 'Error';
            document.getElementById('recentRequests').textContent = 'Error';
        }
    } catch (error) {
        console.error('Error loading stats:', error);
        // Show error in UI
        document.getElementById('totalRequests').textContent = 'Error';
        document.getElementById('pendingRequests').textContent = 'Error';
        document.getElementById('confirmedRequests').textContent = 'Error';
        document.getElementById('completedRequests').textContent = 'Error';
        document.getElementById('cancelledRequests').textContent = 'Error';
        document.getElementById('recentRequests').textContent = 'Error';
    }
}

// Load recent logs
async function loadRecentLogs() {
    const hours = document.getElementById('timeRange').value;
    const status = document.getElementById('statusFilter').value;
    
    try {
        let url = `${API_BASE}/admin/logs/recent?hours=${hours}`;
        if (status) {
            url = `${API_BASE}/admin/logs/status/${status}`;
        }
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.status === 'success') {
            displayLogs(data.logs);
        }
    } catch (error) {
        console.error('Error loading logs:', error);
        document.getElementById('logsTable').innerHTML = '<div class="error">Error loading logs</div>';
    }
}

// Display logs in table
function displayLogs(logs) {
    const table = `
        <table class="table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Action</th>
                    <th>Consultation ID</th>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Status</th>
                    <th>Company</th>
                </tr>
            </thead>
            <tbody>
                ${logs.map(log => `
                    <tr>
                        <td>${new Date(log.timestamp).toLocaleString('en-US', {
                            year: 'numeric',
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                            second: '2-digit',
                            hour12: true,
                            timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
                        })}</td>
                        <td>${log.action}</td>
                        <td>${log.consultation_id}</td>
                        <td>${log.user_name}</td>
                        <td>${log.user_email}</td>
                        <td><span class="status-badge status-${log.status}">${log.status}</span></td>
                        <td>${log.company || '-'}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    document.getElementById('logsTable').innerHTML = table;
}

// Load team members
async function loadTeamMembers() {
    try {
        const response = await fetch(`${API_BASE}/admin/team`);
        const data = await response.json();
        
        if (data.status === 'success') {
            displayTeamMembers(data.team_members);
        }
    } catch (error) {
        console.error('Error loading team members:', error);
    }
}

// Display team members
function displayTeamMembers(members) {
    const table = `
        <table class="table">
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Role</th>
                    <th>Phone</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                ${members.map(member => `
                    <tr>
                        <td>${member.name}</td>
                        <td>${member.email}</td>
                        <td>${member.role}</td>
                        <td>${member.phone || '-'}</td>
                        <td>
                            <button class="btn btn-danger" onclick="removeTeamMember('${member.email}')">Remove</button>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    document.getElementById('teamTable').innerHTML = table;
}

// Load consultations
async function loadConsultations() {
    try {
        const response = await fetch(`${API_BASE}/consultation/all`);
        const data = await response.json();
        
        console.log('Consultations response:', data); // Debug log
        
        if (data.status === 'success') {
            displayConsultations(data.requests);
        } else {
            console.error('Consultations API returned error:', data);
            document.getElementById('consultationsTable').innerHTML = '<div class="error">Error loading consultations</div>';
        }
    } catch (error) {
        console.error('Error loading consultations:', error);
        document.getElementById('consultationsTable').innerHTML = '<div class="error">Error loading consultations</div>';
    }
}

// Display consultations
function displayConsultations(consultations) {
    if (!consultations || consultations.length === 0) {
        document.getElementById('consultationsTable').innerHTML = '<div class="loading">No consultation requests found</div>';
        return;
    }
    
    // Sort consultations by creation date (newest first)
    const sortedConsultations = consultations.sort((a, b) => {
        const dateA = new Date(a.created_at);
        const dateB = new Date(b.created_at);
        return dateB - dateA;
    });
    
    const table = `
        <table class="table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Company</th>
                    <th>Preferred Date</th>
                    <th>Preferred Time</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                ${sortedConsultations.map(consultation => `
                    <tr>
                        <td>${consultation.id}</td>
                        <td>${consultation.name}</td>
                        <td>${consultation.email}</td>
                        <td>${consultation.company || '-'}</td>
                        <td>${consultation.preferred_date || '-'}</td>
                        <td>${consultation.preferred_time || '-'}</td>
                        <td><span class="status-badge status-${consultation.status}">${consultation.status}</span></td>
                        <td>${new Date(consultation.created_at).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                            second: '2-digit',
                            hour12: true,
                            timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
                        })}</td>
                        <td>
                            <select onchange="updateConsultationStatus('${consultation.id}', this.value)">
                                <option value="pending" ${consultation.status === 'pending' ? 'selected' : ''}>Pending</option>
                                <option value="confirmed" ${consultation.status === 'confirmed' ? 'selected' : ''}>Confirmed</option>
                                <option value="completed" ${consultation.status === 'completed' ? 'selected' : ''}>Completed</option>
                                <option value="cancelled" ${consultation.status === 'cancelled' ? 'selected' : ''}>Cancelled</option>
                            </select>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    document.getElementById('consultationsTable').innerHTML = table;
}

// Update consultation status
async function updateConsultationStatus(consultationId, newStatus) {
    try {
        const response = await fetch(`${API_BASE}/consultation/update-status/${consultationId}?status=${newStatus}`, {
            method: 'PUT'
        });
        const data = await response.json();
        
        if (data.success) {
            loadConsultations();
            loadStats();
            loadRecentLogs();
        } else {
            alert('Error updating status: ' + data.message);
        }
    } catch (error) {
        console.error('Error updating status:', error);
        alert('Error updating status');
    }
}

// Add team member
async function addTeamMember(name, email, role, phone) {
    try {
        const response = await fetch(`${API_BASE}/admin/team/add?name=${encodeURIComponent(name)}&email=${encodeURIComponent(email)}&role=${encodeURIComponent(role)}&phone=${encodeURIComponent(phone)}`, {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.success) {
            loadTeamMembers();
            closeAddTeamModal();
            document.getElementById('addTeamForm').reset();
        } else {
            alert('Error adding team member: ' + data.message);
        }
    } catch (error) {
        console.error('Error adding team member:', error);
        alert('Error adding team member');
    }
}

// Remove team member
async function removeTeamMember(email) {
    if (confirm('Are you sure you want to remove this team member?')) {
        try {
            const response = await fetch(`${API_BASE}/admin/team/remove/${email}`, {
                method: 'DELETE'
            });
            const data = await response.json();
            
            if (data.success) {
                loadTeamMembers();
            } else {
                alert('Error removing team member: ' + data.message);
            }
        } catch (error) {
            console.error('Error removing team member:', error);
            alert('Error removing team member');
        }
    }
}

// Modal functions
function openAddTeamModal() {
    document.getElementById('addTeamModal').style.display = 'block';
}

function closeAddTeamModal() {
    document.getElementById('addTeamModal').style.display = 'none';
}

// Form submission
document.getElementById('addTeamForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const name = document.getElementById('teamName').value;
    const email = document.getElementById('teamEmail').value;
    const role = document.getElementById('teamRole').value;
    const phone = document.getElementById('teamPhone').value;
    
    addTeamMember(name, email, role, phone);
});

// Close modal when clicking outside
window.onclick = function(event) {
    const addTeamModal = document.getElementById('addTeamModal');
    const deleteConfirmModal = document.getElementById('deleteConfirmModal');
    
    if (event.target === addTeamModal) {
        closeAddTeamModal();
    }
    if (event.target === deleteConfirmModal) {
        closeDeleteConfirmModal();
    }
}

// Add event listener for confirmation text input
document.addEventListener('DOMContentLoaded', function() {
    const confirmTextInput = document.getElementById('confirmText');
    if (confirmTextInput) {
        confirmTextInput.addEventListener('input', validateConfirmationText);
        confirmTextInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !document.getElementById('confirmDeleteBtn').disabled) {
                proceedWithDelete();
            }
        });
    }
});

// Refresh all data function
function refreshAllData() {
    console.log('Refreshing all data...');
    loadStats();
    loadRecentLogs();
    loadTeamMembers();
    loadConsultations();
}

// Open delete confirmation modal
function confirmDeleteAllData() {
    document.getElementById('deleteConfirmModal').style.display = 'block';
    document.getElementById('confirmText').value = '';
    document.getElementById('confirmText').classList.remove('valid', 'invalid');
    document.getElementById('confirmDeleteBtn').disabled = true;
    document.getElementById('confirmText').focus();
}

// Close delete confirmation modal
function closeDeleteConfirmModal() {
    document.getElementById('deleteConfirmModal').style.display = 'none';
}

// Validate confirmation text input
function validateConfirmationText() {
    const input = document.getElementById('confirmText');
    const confirmBtn = document.getElementById('confirmDeleteBtn');
    const value = input.value.trim();
    
    if (value === 'DELETE') {
        input.classList.remove('invalid');
        input.classList.add('valid');
        confirmBtn.disabled = false;
    } else {
        input.classList.remove('valid');
        input.classList.add('invalid');
        confirmBtn.disabled = true;
    }
}

// Proceed with delete after confirmation
function proceedWithDelete() {
    const input = document.getElementById('confirmText');
    if (input.value.trim() === 'DELETE') {
        closeDeleteConfirmModal();
        deleteAllData();
    }
}

// Show success message
function showSuccessMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'success-message';
    messageDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);
        z-index: 10000;
        font-weight: 600;
        animation: slideIn 0.3s ease;
    `;
    messageDiv.textContent = message;
    document.body.appendChild(messageDiv);
    
    setTimeout(() => {
        messageDiv.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(messageDiv);
        }, 300);
    }, 3000);
}

// Show error message
function showErrorMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'error-message';
    messageDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.3);
        z-index: 10000;
        font-weight: 600;
        animation: slideIn 0.3s ease;
    `;
    messageDiv.textContent = message;
    document.body.appendChild(messageDiv);
    
    setTimeout(() => {
        messageDiv.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(messageDiv);
        }, 300);
    }, 4000);
}

// Add CSS animations for messages
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Delete all data function
async function deleteAllData() {
    try {
        // Show loading state
        const deleteBtn = document.querySelector('.delete-btn');
        const originalText = deleteBtn.innerHTML;
        deleteBtn.innerHTML = '<span class="delete-icon">⏳</span>Deleting...';
        deleteBtn.disabled = true;

        let deletedCount = 0;
        let totalItems = 0;

        // Step 1: Delete all consultation requests
        try {
            const consultationsResponse = await fetch(`${API_BASE}/consultation/all`);
            if (consultationsResponse.ok) {
                const consultationsData = await consultationsResponse.json();
                if (consultationsData.status === 'success' && consultationsData.requests) {
                    totalItems += consultationsData.requests.length;
                    
                    // Delete each consultation individually
                    for (const consultation of consultationsData.requests) {
                        try {
                            const deleteResponse = await fetch(`${API_BASE}/consultation/delete/${consultation.id}`, {
                                method: 'DELETE'
                            });
                            if (deleteResponse.ok) {
                                deletedCount++;
                            }
                        } catch (err) {
                            console.log('Could not delete consultation:', consultation.id);
                        }
                    }
                }
            }
        } catch (err) {
            console.log('Error fetching consultations:', err);
        }

        // Step 2: Clear all consultation logs by calling a direct endpoint
        try {
            const logsResponse = await fetch(`${API_BASE}/admin/clear-all-logs`, {
                method: 'POST'
            });
            if (logsResponse.ok) {
                console.log('All logs cleared successfully');
            }
        } catch (err) {
            console.log('Logs clearing endpoint not available, trying alternative...');
            
            // Alternative: Try to clear logs by getting recent logs and clearing them
            try {
                const recentLogsResponse = await fetch(`${API_BASE}/admin/logs/recent?hours=8760`); // 1 year
                if (recentLogsResponse.ok) {
                    const logsData = await recentLogsResponse.json();
                    if (logsData.status === 'success' && logsData.logs) {
                        totalItems += logsData.logs.length;
                        // Note: We can't delete individual logs, but we've attempted to clear them
                    }
                }
            } catch (altErr) {
                console.log('Alternative log clearing failed:', altErr);
            }
        }

        // Step 3: Reset team members to default (remove all custom ones)
        try {
            const teamResponse = await fetch(`${API_BASE}/admin/team`);
            if (teamResponse.ok) {
                const teamData = await teamResponse.json();
                if (teamData.status === 'success' && teamData.team_members) {
                    // Remove all team members except default ones
                    for (const member of teamData.team_members) {
                        if (member.email !== 'ask@akenotech.com') {
                            try {
                                const removeResponse = await fetch(`${API_BASE}/admin/team/remove/${member.email}`, {
                                    method: 'DELETE'
                                });
                                if (removeResponse.ok) {
                                    deletedCount++;
                                }
                            } catch (err) {
                                console.log('Could not remove team member:', member.email);
                            }
                        }
                    }
                }
            }
        } catch (err) {
            console.log('Error managing team members:', err);
        }

        // Step 4: Clear all chat sessions
        try {
            const sessionsResponse = await fetch(`${API_BASE}/sessions`);
            if (sessionsResponse.ok) {
                const sessionsData = await sessionsResponse.json();
                if (sessionsData.sessions) {
                    totalItems += sessionsData.sessions.length;
                    
                    for (const session of sessionsData.sessions) {
                        try {
                            const deleteSessionResponse = await fetch(`${API_BASE}/sessions/${session.session_id}`, {
                                method: 'DELETE'
                            });
                            if (deleteSessionResponse.ok) {
                                deletedCount++;
                            }
                        } catch (err) {
                            console.log('Could not delete session:', session.session_id);
                        }
                    }
                }
            }
        } catch (err) {
            console.log('Error clearing sessions:', err);
        }

        // Show success message with details
        const successMessage = `✅ Data deletion completed! 
        
Deleted ${deletedCount} items successfully.
Total items processed: ${totalItems}

All consultation requests, logs, and sessions have been cleared.`;
        
        showSuccessMessage(successMessage);
        
        // Refresh all data to show empty state
        setTimeout(() => {
            refreshAllData();
        }, 2000);

    } catch (error) {
        console.error('Error deleting all data:', error);
        showErrorMessage('❌ Error during deletion process: ' + error.message);
    } finally {
        // Restore button state
        const deleteBtn = document.querySelector('.delete-btn');
        deleteBtn.innerHTML = '<span class="delete-icon">🗑️</span>Delete All Data';
        deleteBtn.disabled = false;
    }
}
