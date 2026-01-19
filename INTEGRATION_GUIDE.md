# Website Integration Guide

This guide provides detailed instructions for integrating the Agentic AI Chatbot into your company website.

## Integration Methods

### 1. Widget Integration (Recommended)

The widget integration is the easiest and most flexible method. It adds a floating chat button to your website that opens a chat interface.

#### Basic Integration

Add this single line to your website's HTML:

```html
<script 
  src="http://your-chatbot-domain.com/static/chatbot-widget.js"
  data-api-url="http://your-chatbot-domain.com/chat"
></script>
```

#### Advanced Configuration

```html
<script 
  src="http://your-chatbot-domain.com/static/chatbot-widget.js"
  data-api-url="http://your-chatbot-domain.com/chat"
  data-position="bottom-right"
  data-primary-color="#4f46e5"
  data-secondary-color="#7c3aed"
  data-title="Agentic AI Assistant"
  data-subtitle="Ask me about agentic AI"
  data-placeholder="Ask me about agentic AI..."
></script>
```

#### Configuration Options

| Attribute | Description | Default | Options |
|-----------|-------------|---------|---------|
| `data-api-url` | Chatbot API endpoint | `/chat` | Any valid URL |
| `data-position` | Widget position | `bottom-right` | `bottom-right`, `bottom-left`, `top-right`, `top-left` |
| `data-primary-color` | Primary theme color | `#4f46e5` | Any valid CSS color |
| `data-secondary-color` | Secondary theme color | `#7c3aed` | Any valid CSS color |
| `data-title` | Chatbot title | `Agentic AI Assistant` | Any string |
| `data-subtitle` | Chatbot subtitle | `Ask me about agentic AI` | Any string |
| `data-placeholder` | Input placeholder | `Ask me about agentic AI...` | Any string |

#### Programmatic Control

```javascript
// Access the chatbot instance
const chatbot = window.agenticAIChatbot;

// Control methods
chatbot.openChat();           // Open the chat interface
chatbot.closeChat();          // Close the chat interface
chatbot.sendMessage('Hello'); // Send a message programmatically
chatbot.clearMessages();      // Clear chat history
chatbot.getMessages();        // Get all messages

// Example: Open chat when user clicks a button
document.getElementById('chat-button').addEventListener('click', () => {
  chatbot.openChat();
});
```

### 2. Iframe Integration

For a more embedded experience, use iframe integration:

```html
<iframe 
  src="http://your-chatbot-domain.com/chat-interface" 
  width="400" 
  height="600"
  frameborder="0"
  style="border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
</iframe>
```

#### Responsive Iframe

```html
<div style="position: relative; width: 100%; height: 600px;">
  <iframe 
    src="http://your-chatbot-domain.com/chat-interface" 
    width="100%" 
    height="100%"
    frameborder="0"
    style="border-radius: 12px;">
  </iframe>
</div>
```

### 3. Custom Integration

For complete control, integrate directly with the API:

```javascript
class CustomChatbot {
  constructor(apiUrl) {
    this.apiUrl = apiUrl;
    this.sessionId = this.generateSessionId();
  }
  
  generateSessionId() {
    return 'custom_' + Math.random().toString(36).substr(2, 9);
  }
  
  async sendMessage(message) {
    try {
      const response = await fetch(this.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          session_id: this.sessionId
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error:', error);
      throw error;
    }
  }
}

// Usage
const chatbot = new CustomChatbot('http://your-chatbot-domain.com/chat');
chatbot.sendMessage('What is agentic AI?')
  .then(response => console.log(response.response))
  .catch(error => console.error(error));
```

## Styling and Customization

### CSS Customization

The widget automatically injects its styles, but you can override them:

```css
/* Customize the chat button */
#agentic-ai-chatbot-widget .chatbot-toggle {
  background: linear-gradient(135deg, #your-color-1, #your-color-2) !important;
}

/* Customize the chat container */
#agentic-ai-chatbot-widget .chatbot-container {
  border-radius: 20px !important;
  box-shadow: 0 10px 40px rgba(0,0,0,0.2) !important;
}

/* Customize message bubbles */
#agentic-ai-chatbot-widget .message.user .message-content {
  background: linear-gradient(135deg, #your-color-1, #your-color-2) !important;
}
```

### Theme Customization

```javascript
// Initialize with custom theme
const chatbot = new AgenticAIChatbotWidget({
  apiUrl: 'http://your-chatbot-domain.com/chat',
  primaryColor: '#your-primary-color',
  secondaryColor: '#your-secondary-color',
  title: 'Your Custom Title',
  subtitle: 'Your Custom Subtitle'
});
```

## Security Considerations

### CORS Configuration

Ensure your chatbot server allows requests from your domain:

```python
# In main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "https://www.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### API Key Protection

Never expose API keys in client-side code. The chatbot API should handle all external API calls server-side.

### Content Security Policy

If you use CSP, ensure it allows the chatbot scripts:

```html
<meta http-equiv="Content-Security-Policy" 
      content="script-src 'self' http://your-chatbot-domain.com; 
               connect-src 'self' http://your-chatbot-domain.com;">
```

## Performance Optimization

### Lazy Loading

Load the chatbot widget only when needed:

```javascript
// Load chatbot when user scrolls or interacts
function loadChatbot() {
  if (!window.agenticAIChatbot) {
    const script = document.createElement('script');
    script.src = 'http://your-chatbot-domain.com/static/chatbot-widget.js';
    script.dataset.apiUrl = 'http://your-chatbot-domain.com/chat';
    document.head.appendChild(script);
  }
}

// Load on scroll
window.addEventListener('scroll', loadChatbot, { once: true });

// Or load on user interaction
document.addEventListener('click', loadChatbot, { once: true });
```

### Caching

Set appropriate cache headers for the widget script:

```python
# In main.py
@app.get("/static/chatbot-widget.js")
async def get_widget_script():
    return FileResponse(
        "static/chatbot-widget.js",
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=3600"}
    )
```

## Analytics and Tracking

### Google Analytics Integration

```javascript
// Track chatbot interactions
const chatbot = window.agenticAIChatbot;

// Track when chat is opened
chatbot.addEventListener('open', () => {
  gtag('event', 'chatbot_open', {
    event_category: 'engagement',
    event_label: 'chatbot'
  });
});

// Track messages sent
chatbot.addEventListener('message_sent', (event) => {
  gtag('event', 'chatbot_message', {
    event_category: 'engagement',
    event_label: 'message_sent',
    value: event.detail.message.length
  });
});
```

### Custom Analytics

```javascript
// Custom event tracking
function trackChatbotEvent(eventName, data) {
  // Send to your analytics service
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

// Track chatbot usage
chatbot.addEventListener('open', () => {
  trackChatbotEvent('chatbot_opened', { session_id: chatbot.sessionId });
});
```

## Testing

### Unit Testing

```javascript
// Test chatbot functionality
describe('Chatbot Integration', () => {
  test('should initialize correctly', () => {
    const chatbot = new AgenticAIChatbotWidget({
      apiUrl: 'http://localhost:8000/chat'
    });
    expect(chatbot).toBeDefined();
    expect(chatbot.sessionId).toBeDefined();
  });
  
  test('should send messages', async () => {
    const chatbot = new AgenticAIChatbotWidget({
      apiUrl: 'http://localhost:8000/chat'
    });
    
    const response = await chatbot.sendMessage('Hello');
    expect(response.response).toBeDefined();
  });
});
```

### Integration Testing

```javascript
// Test with real API
async function testChatbotIntegration() {
  const chatbot = new AgenticAIChatbotWidget({
    apiUrl: 'http://your-chatbot-domain.com/chat'
  });
  
  try {
    const response = await chatbot.sendMessage('What is agentic AI?');
    console.log('✅ Chatbot integration working:', response.response);
  } catch (error) {
    console.error('❌ Chatbot integration failed:', error);
  }
}
```

## Troubleshooting

### Common Issues

1. **Widget not loading**: Check the script URL and network connectivity
2. **CORS errors**: Ensure your chatbot server allows your domain
3. **API errors**: Check the API endpoint and authentication
4. **Styling issues**: Verify CSS overrides and theme configuration

### Debug Mode

Enable debug mode for detailed logging:

```javascript
const chatbot = new AgenticAIChatbotWidget({
  apiUrl: 'http://your-chatbot-domain.com/chat',
  debug: true  // Enable debug logging
});
```

### Network Debugging

```javascript
// Monitor network requests
const originalFetch = window.fetch;
window.fetch = function(...args) {
  console.log('Fetch request:', args);
  return originalFetch.apply(this, args)
    .then(response => {
      console.log('Fetch response:', response);
      return response;
    });
};
```

## Best Practices

1. **Load the widget asynchronously** to avoid blocking page load
2. **Use HTTPS** for all chatbot communications
3. **Implement proper error handling** for network failures
4. **Test across different browsers** and devices
5. **Monitor performance** and optimize as needed
6. **Keep the chatbot updated** with the latest version
7. **Provide fallback options** for users with JavaScript disabled

## Support

For integration support:
- Check the console for error messages
- Verify API endpoints are accessible
- Test with the provided examples
- Contact the development team for assistance

---

**Happy integrating! 🚀**
