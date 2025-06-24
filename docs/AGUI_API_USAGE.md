# 📚 AG-UI API Usage Guide

Panduan lengkap untuk menggunakan AG-UI API endpoints dengan format request yang benar.

## 🌐 Base URL
```
https://api.reserse.id/v1/agui
```

## 🔑 Authentication
Semua request memerlukan API key di header:
```
X-API-Key: your-api-key-here
```

## 📋 Available Endpoints

### 1. **List Available Agents**
```http
GET /v1/agui
```

**Response:**
```json
["tipidter"]
```

---

### 2. **Get Agent Information**
```http
GET /v1/agui/{agent_id}/info
```

**Example:**
```bash
curl -X GET "https://api.reserse.id/v1/agui/tipidter/info" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "agent_id": "tipidter-agui",
  "name": "Penyidik Tindak Pidana Tertentu (Tipidter) AG-UI",
  "description": "Asisten penyidik kepolisian untuk tindak pidana tertentu",
  "status": "active",
  "app_id": "tipidter_agui",
  "endpoints": {
    "agui_protocol": "/api/v1/agui/tipidter/agui",
    "chat": "/api/v1/agui/tipidter/chat",
    "info": "/api/v1/agui/tipidter/info",
    "health": "/api/v1/agui/tipidter/health"
  }
}
```

---

### 3. **Health Check**
```http
GET /v1/agui/{agent_id}/health
```

**Example:**
```bash
curl -X GET "https://api.reserse.id/v1/agui/tipidter/health" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "status": "healthy",
  "agent_id": "tipidter-agui",
  "message": "AG-UI agent tipidter is running properly"
}
```

---

### 4. **Direct Chat** ⭐
```http
POST /v1/agui/{agent_id}/chat
```

**Request Format:**
```json
{
  "message": "Your question here",
  "user_id": "optional-user-id",
  "session_id": "optional-session-id",
  "stream": false
}
```

**Example Request:**
```bash
curl -X POST "https://api.reserse.id/v1/agui/tipidter/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "Halo, saya butuh bantuan tentang kasus pembalakan liar di hutan lindung",
    "user_id": "user123",
    "session_id": "session456",
    "stream": false
  }'
```

**Response:**
```json
{
  "response": "Berdasarkan pencarian knowledge base, kasus pembalakan liar di hutan lindung merupakan tindak pidana yang diatur dalam beberapa peraturan...",
  "status": "success",
  "agent_id": "tipidter-agui",
  "session_id": "session456"
}
```

---

### 5. **AG-UI Protocol** (For Frontend Integration)
```http
POST /v1/agui/{agent_id}/v1/agui
```

Endpoint ini mengikuti [AG-UI Protocol specification](https://docs.agno.com/applications/ag-ui/introduction) dan digunakan oleh:
- Dojo frontend
- CopilotKit integration
- Custom AG-UI clients

---

## 🛠️ Request Examples

### **Minimal Request**
```json
{
  "message": "Apa itu pembalakan liar?"
}
```

### **Full Request with Session**
```json
{
  "message": "Bagaimana proses penyidikan kasus pembalakan liar?",
  "user_id": "penyidik_001",
  "session_id": "investigation_session_123",
  "stream": false
}
```

### **Streaming Request**
```json
{
  "message": "Jelaskan unsur-unsur tindak pidana kehutanan",
  "user_id": "user123",
  "session_id": "session456",
  "stream": true
}
```

## ❌ Common Errors

### **400 Bad Request - Empty Message**
```json
{
  "detail": "Message cannot be empty"
}
```

**Cause:** Request body tidak mengandung `message` atau `message` kosong.

**Solution:**
```json
{
  "message": "Your actual question here"
}
```

### **400 Bad Request - Invalid JSON**
```json
{
  "detail": "Invalid JSON format"
}
```

**Cause:** Format JSON tidak valid.

**Solution:** Pastikan JSON format benar dan gunakan `Content-Type: application/json`.

### **401 Unauthorized**
```json
{
  "detail": "Invalid API key"
}
```

**Cause:** API key tidak valid atau tidak disertakan.

**Solution:** Tambahkan header `X-API-Key: your-valid-api-key`.

### **404 Not Found**
```json
{
  "detail": "Agent not found: tipidter2"
}
```

**Cause:** Agent ID tidak valid.

**Solution:** Gunakan agent ID yang valid (lihat endpoint `/v1/agui` untuk list agents).

## 🧪 Testing with Different Tools

### **cURL**
```bash
curl -X POST "https://api.reserse.id/v1/agui/tipidter/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"message": "Test message"}'
```

### **Python requests**
```python
import requests

url = "https://api.reserse.id/v1/agui/tipidter/chat"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key"
}
data = {
    "message": "Halo, saya butuh bantuan tentang kasus tipidter",
    "user_id": "user123"
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### **JavaScript/Node.js**
```javascript
const response = await fetch('https://api.reserse.id/v1/agui/tipidter/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-api-key'
  },
  body: JSON.stringify({
    message: 'Bagaimana menangani kasus pembalakan liar?',
    user_id: 'user123',
    session_id: 'session456'
  })
});

const result = await response.json();
console.log(result);
```

## 📝 Notes

1. **Message Field:** Wajib diisi dan tidak boleh kosong
2. **User ID & Session ID:** Opsional, berguna untuk tracking dan session management
3. **Streaming:** Set `stream: true` untuk response streaming (jika didukung client)
4. **Content-Type:** Harus `application/json`
5. **API Key:** Wajib disertakan di header untuk authentication

---

**Happy coding! 🚀** 