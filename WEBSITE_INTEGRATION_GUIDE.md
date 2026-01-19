# Website Integration Guide - Hostinger Style

## Overview

This guide shows you how to integrate your agentic AI chatbot into your company website exactly like the Hostinger example you showed. The chatbot will appear as a floating widget that users can click to open a chat interface.

## 🎯 **Integration Methods**

### **Method 1: Simple Widget Integration (Recommended)**

Add this single line to your website's HTML:

```html
<script src="http://your-domain.com/static/chatbot-widget.js"
        data-api-url="http://your-domain.com/chat"
        data-position="bottom-right"
        data-primary-color="#4f46e5"
        data-secondary-color="#7c3aed"
        data-title="AI Sales Expert"
        data-subtitle="Hi! I'm your AI sales expert. How can I help you today?"
        data-placeholder="How can agentic AI help my business?">
</script>
```

### **Method 2: Custom Styling**

For more control over appearance:

```html
<script src="http://your-domain.com/static/chatbot-widget.js"
        data-api-url="http://your-domain.com/chat"
        data-position="bottom-right"
        data-primary-color="#your-brand-color"
        data-secondary-color="#your-secondary-color"
        data-title="Your Custom Title"
        data-subtitle="Your custom greeting"
        data-placeholder="Your custom placeholder">
</script>
```

## 🎨 **Customization Options**

### **Position Options:**
- `bottom-right` (default)
- `bottom-left`
- `top-right`
- `top-left`

### **Color Customization:**
- `data-primary-color` - Main brand color
- `data-secondary-color` - Secondary color for gradients

### **Text Customization:**
- `data-title` - Chatbot window title
- `data-subtitle` - Greeting message
- `data-placeholder` - Input field placeholder

## 📱 **Responsive Design**

The widget automatically adapts to different screen sizes:
- **Desktop**: Full-size chat window
- **Mobile**: Optimized for touch interaction
- **Tablet**: Medium-sized interface

## 🔧 **Technical Requirements**

### **Server Setup:**
1. Deploy your chatbot to a web server
2. Ensure CORS is configured for your domain
3. Use HTTPS for production

### **CORS Configuration:**
Update your `main.py` to allow your domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "https://www.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 🚀 **Production Deployment**

### **Step 1: Deploy Chatbot**
```bash
# Deploy to your server
python main.py
```

### **Step 2: Update Domain**
Replace `localhost:8000` with your actual domain:

```html
<script src="https://your-domain.com/static/chatbot-widget.js"
        data-api-url="https://your-domain.com/chat">
</script>
```

### **Step 3: Test Integration**
1. Open your website
2. Look for the floating chat button
3. Click to open the chat interface
4. Test with sample questions

## 📊 **Analytics & Tracking**

### **Google Analytics Integration:**
```javascript
// Track chatbot interactions
const chatbot = window.agenticAIChatbot;

chatbot.addEventListener('open', () => {
  gtag('event', 'chatbot_open', {
    event_category: 'engagement',
    event_label: 'chatbot'
  });
});
```

### **Custom Analytics:**
```javascript
// Track specific interactions
function trackChatbotEvent(eventName, data) {
  fetch('/api/analytics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event: eventName,
      data: data,
      timestamp: new Date().toISOString()
    })
  });
}
```

## 🎯 **Sales Optimization**

### **Lead Capture:**
The chatbot automatically:
- Identifies potential customers
- Captures contact information
- Schedules consultations
- Tracks conversion metrics

### **Call-to-Actions:**
Every response includes:
- "Schedule a free consultation"
- "Book a demo"
- "Contact our team"
- "Get a personalized quote"

## 🔒 **Security Considerations**

### **Content Security Policy:**
```html
<meta http-equiv="Content-Security-Policy" 
      content="script-src 'self' https://your-domain.com; 
               connect-src 'self' https://your-domain.com;">
```

### **API Security:**
- Use HTTPS in production
- Implement rate limiting
- Validate all inputs
- Monitor for abuse

## 📈 **Performance Optimization**

### **Lazy Loading:**
```javascript
// Load chatbot only when needed
function loadChatbot() {
  if (!window.agenticAIChatbot) {
    const script = document.createElement('script');
    script.src = 'https://your-domain.com/static/chatbot-widget.js';
    script.dataset.apiUrl = 'https://your-domain.com/chat';
    document.head.appendChild(script);
  }
}

// Load on user interaction
document.addEventListener('click', loadChatbot, { once: true });
```

### **Caching:**
Set appropriate cache headers for the widget script to improve loading times.

## 🧪 **Testing Checklist**

- [ ] Widget loads on your website
- [ ] Chat interface opens when clicked
- [ ] Messages send and receive responses
- [ ] Responses are sales-focused and formatted cleanly
- [ ] Call-to-actions are present
- [ ] Mobile responsiveness works
- [ ] Analytics tracking functions
- [ ] Performance is acceptable

## 🎉 **Example Integration**

See `website_integration_example.html` for a complete example of how to integrate the chatbot into a company website.

## 📞 **Support**

For integration support:
- Check the console for error messages
- Verify API endpoints are accessible
- Test with the provided examples
- Contact the development team for assistance

---

**Your chatbot is now ready to generate leads and drive sales on your website!** 🚀
