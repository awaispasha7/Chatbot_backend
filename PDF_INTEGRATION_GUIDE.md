# PDF Integration Guide

## Overview

This guide shows you how to add your company's PDF documents (projects, proposals, case studies) to the chatbot's knowledge base for more accurate and company-specific responses.

## 🚀 **Quick Start**

### **Method 1: Web Interface (Easiest)**

1. **Start your chatbot:**
   ```bash
   python run.py
   ```

2. **Open the PDF upload page:**
   ```
   http://localhost:8000/pdf-upload
   ```

3. **Upload your PDF:**
   - Drag and drop your PDF file
   - Or click to select your PDF
   - Click "Upload to Knowledge Base"

4. **Test the chatbot:**
   - Go to `http://localhost:8000/chat-interface`
   - Ask questions about your projects and proposals

### **Method 2: Command Line**

1. **Install PyPDF2:**
   ```bash
   pip install PyPDF2
   ```

2. **Add your PDF:**
   ```bash
   python add_pdf.py your_company_projects.pdf
   ```

3. **Test the chatbot:**
   - Ask questions about your specific projects

### **Method 3: API Upload**

```bash
curl -X POST "http://localhost:8000/knowledge-base/upload-pdf" \
     -F "file=@your_company_projects.pdf"
```

## 📄 **What PDFs Should You Upload?**

### **Recommended Documents:**
- ✅ **Recent Project Proposals** - Show your capabilities
- ✅ **Case Studies** - Demonstrate success stories
- ✅ **Technical Specifications** - Detailed project information
- ✅ **Company Portfolio** - Overview of your work
- ✅ **Service Descriptions** - Detailed service offerings
- ✅ **Client Testimonials** - Social proof and results

### **Example Questions After Upload:**
- "What recent projects have you completed?"
- "Can you tell me about your AI implementation case studies?"
- "What are your technical capabilities?"
- "Show me examples of your work"
- "What results have you achieved for clients?"

## 🔧 **Technical Details**

### **PDF Processing:**
- **Text Extraction**: Uses PyPDF2 to extract text from PDFs
- **Chunking**: Splits large documents into manageable chunks
- **Vector Storage**: Converts text to embeddings for semantic search
- **Metadata**: Tags documents as "company_document" for easy identification
- **Persistence**: All data is saved to disk and persists after restart

### **Data Persistence:**
- ✅ **Automatic Saving**: PDFs are automatically saved to disk
- ✅ **Restart Safe**: Data persists after stopping and restarting the chatbot
- ✅ **No Re-upload**: Once uploaded, PDFs stay in the knowledge base
- ✅ **File Storage**: Data stored in `./chroma_db` directory

### **File Requirements:**
- **Format**: PDF files only (.pdf extension)
- **Size**: No specific limit, but larger files take longer to process
- **Content**: Text-based PDFs work best (scanned images may not work well)

## 📊 **Testing Your PDF Integration**

### **Test Questions to Ask:**

**About Projects:**
- "What projects have you completed recently?"
- "Can you describe your most successful AI implementation?"
- "What industries do you work with?"

**About Capabilities:**
- "What technical skills does your team have?"
- "What AI technologies do you use?"
- "What's your development process?"

**About Results:**
- "What results have you achieved for clients?"
- "Can you share any case studies?"
- "What ROI have your clients seen?"

### **Expected Improvements:**
- ✅ **More Specific Responses** - References your actual projects
- ✅ **Company-Specific Examples** - Uses your real case studies
- ✅ **Accurate Capabilities** - Based on your actual work
- ✅ **Better Lead Generation** - More relevant to your services

## 🎯 **Sales Benefits**

### **Before PDF Integration:**
- Generic responses about agentic AI
- No specific examples
- Limited company context

### **After PDF Integration:**
- References your actual projects
- Shares real case studies
- Demonstrates your expertise
- Builds credibility with prospects

## 🔍 **Troubleshooting**

### **Common Issues:**

**PDF Not Processing:**
- Check if PDF contains text (not just images)
- Ensure file is not corrupted
- Try with a smaller PDF first

**Poor Text Quality:**
- Use text-based PDFs, not scanned images
- Ensure good OCR if using scanned documents
- Check PDF text extraction manually

**Responses Not Improved:**
- Wait a few minutes for processing to complete
- Try asking more specific questions
- Check if PDF content is relevant to questions

### **Debug Commands:**
```bash
# Check if PDF was added
curl "http://localhost:8000/knowledge-base/search?query=your_project_name"

# View all documents
curl "http://localhost:8000/knowledge-base/search?query=company_document"

# Check knowledge base status
curl "http://localhost:8000/knowledge-base/status"

# Test persistence
python test_persistence.py
```

### **Verify Persistence:**
1. **Upload a PDF** using any method
2. **Stop the chatbot** (Ctrl+C)
3. **Restart the chatbot** (`python run.py`)
4. **Check status**: Visit `http://localhost:8000/knowledge-base/status`
5. **Test search**: Ask questions about your uploaded content

**Expected Result**: Your PDF content should still be available without re-uploading!

## 📈 **Best Practices**

### **PDF Preparation:**
1. **Use Text-Based PDFs** - Avoid scanned images
2. **Include Keywords** - Use terms prospects might search for
3. **Structure Content** - Use headings and clear sections
4. **Include Results** - Quantify your achievements
5. **Keep Updated** - Upload new projects regularly

### **Content Optimization:**
- **Project Names** - Use descriptive project titles
- **Technologies** - Mention specific AI technologies used
- **Results** - Include metrics and outcomes
- **Industries** - Specify target industries
- **Services** - Detail your service offerings

## 🚀 **Advanced Usage**

### **Multiple PDFs:**
You can upload multiple PDFs to build a comprehensive knowledge base:
```bash
python add_pdf.py project1.pdf
python add_pdf.py project2.pdf
python add_pdf.py case_studies.pdf
```

### **Regular Updates:**
Set up a process to regularly update your knowledge base with new projects and proposals.

### **Content Management:**
- Organize PDFs by project type
- Include metadata in filenames
- Keep a backup of uploaded documents

## 🎉 **Success Metrics**

After uploading your PDFs, you should see:
- ✅ **More Relevant Responses** - Chatbot references your actual work
- ✅ **Better Lead Quality** - Prospects get specific examples
- ✅ **Increased Engagement** - More detailed conversations
- ✅ **Higher Conversion** - Better sales outcomes

---

**Your chatbot is now powered by your actual company knowledge!** 🚀

Upload your PDFs and watch your chatbot become a true expert on your company's capabilities and achievements.
