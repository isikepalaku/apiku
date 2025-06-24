# 🎯 Implementasi AG-UI Modular untuk Agno Agents

Dokumentasi ini menjelaskan implementasi AG-UI (Agent-User Interaction Protocol) yang modular dan scalable untuk project ini.

## 📋 Struktur Implementasi

### 1. **Agent Definition** (`agents/tipidter_agui.py`)
```python
def get_tipidter_agui_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    # Implementasi agent khusus untuk AG-UI
```

**Karakteristik:**
- ✅ Agent terpisah dari chat biasa dengan database terpisah
- ✅ Memory table: `tipidter_agui_memory`
- ✅ Storage table: `tipidter_agui_storage`
- ✅ Optimized untuk AG-UI interface

### 2. **Operator Pattern** (`agents/agui_operator.py`)
```python
class AGUIAgentType(str, Enum):
    tipidter = "tipidter"
    # Future agents: kuhap, kuhp, narkotika, etc.

def get_agui_agent(agent_id: AGUIAgentType, ...) -> Agent:
    # Factory function untuk mendapatkan agent
```

**Manfaat:**
- ✅ Centralized agent management
- ✅ Type safety dengan Enum
- ✅ Mudah menambahkan agent baru

### 3. **Modular Router** (`api/routes/agui.py`)
```python
def create_agui_app(agent_id: AGUIAgentType) -> AGUIApp:
    # Setup your Agno Agent
    agent = get_agui_agent(agent_id)
    
    # Setup the AG-UI app sesuai dokumentasi
    agui_app = AGUIApp(
        agent=agent,
        name=f"{agent.name} AG-UI",
        app_id=f"{agent_id.value}_agui",
    )
    return agui_app
```

**Fitur:**
- ✅ Dynamic routing: `/{agent_id}/...`
- ✅ AG-UI app caching untuk performa
- ✅ Proper FastAPI sub-app mounting
- ✅ Multiple endpoints per agent

## 🌐 Endpoints yang Tersedia

### **Base URL:** `http://localhost:8000/api/v1/agui`

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/` | List semua AG-UI agents |
| `GET` | `/{agent_id}/info` | Informasi detail agent |
| `GET` | `/{agent_id}/health` | Health check agent |
| `POST` | `/{agent_id}/chat` | Direct chat (non-AG-UI) |
| `POST` | `/{agent_id}/v1/agui` | **AG-UI Protocol Endpoint** |

### **AG-UI Protocol Endpoint**
```
POST /api/v1/agui/tipidter/v1/agui
```

Endpoint ini mengikuti [AG-UI Protocol specification](https://docs.agno.com/applications/ag-ui/introduction) dan compatible dengan:
- **Dojo** frontend
- **CopilotKit** integration
- Custom AG-UI clients

## 🚀 Cara Menambahkan Agent Baru

### Step 1: Buat Agent File
```python
# agents/kuhap_agui.py
def get_kuhap_agui_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="KUHAP Agent AG-UI",
        agent_id="kuhap-agui",
        # ... konfigurasi lainnya
    )
```

### Step 2: Update Operator
```python
# agents/agui_operator.py
class AGUIAgentType(str, Enum):
    tipidter = "tipidter"
    kuhap = "kuhap"  # ← Tambahkan ini

def get_agui_agent(agent_id: AGUIAgentType, ...):
    if agent_id == AGUIAgentType.tipidter:
        return get_tipidter_agui_agent(...)
    elif agent_id == AGUIAgentType.kuhap:  # ← Tambahkan ini
        return get_kuhap_agui_agent(...)
```

### Step 3: Restart Server
```bash
source .venv/bin/activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Otomatis tersedia:**
- ✅ `GET /api/v1/agui/kuhap/info`
- ✅ `POST /api/v1/agui/kuhap/v1/agui`
- ✅ `POST /api/v1/agui/kuhap/chat`

## 🔧 Technical Details

### **AG-UI App Creation**
Mengikuti struktur dokumentasi Agno:
```python
# Setup your Agno Agent, can be any Agno Agent
agent = get_agui_agent(agent_id)

# Setup the AG-UI app
agui_app = AGUIApp(
    agent=agent,
    name="Agent AG-UI",
    app_id="agent_agui",
)

# Get FastAPI app
app = agui_app.get_app()
```

### **FastAPI Sub-App Mounting**
```python
# Mount AG-UI app sebagai sub-application
agui_router.mount(
    f"/{agent_type.value}",
    agui_fastapi_app,
    name=f"agui_{agent_type.value}"
)
```

### **Database Separation**
Setiap AG-UI agent memiliki database terpisah:
- **Memory:** `{agent}_agui_memory`
- **Storage:** `{agent}_agui_storage`
- **Knowledge:** Shared knowledge base per domain

## 🌟 Keunggulan Implementasi

### **1. Modular & Scalable**
- ✅ Mudah menambahkan agent baru
- ✅ Isolated database per agent
- ✅ Type-safe dengan Enum

### **2. Consistent Architecture**
- ✅ Mengikuti pola project yang ada
- ✅ Konsisten dengan `agents.py` pattern
- ✅ Proper error handling & logging

### **3. AG-UI Compliant**
- ✅ Mengikuti dokumentasi Agno
- ✅ Compatible dengan Dojo frontend
- ✅ Support AG-UI protocol specification

### **4. Developer Friendly**
- ✅ Clear separation of concerns
- ✅ Easy to test dan debug
- ✅ Comprehensive documentation

## 🧪 Testing

### **Test Agent Creation**
```bash
python -c "from agents.agui_operator import get_agui_agent, AGUIAgentType; print(get_agui_agent(AGUIAgentType.tipidter))"
```

### **Test API Endpoints**
```bash
# List agents
curl -X GET "http://localhost:8000/api/v1/agui" -H "X-API-Key: your-api-key"

# Agent info
curl -X GET "http://localhost:8000/api/v1/agui/tipidter/info" -H "X-API-Key: your-api-key"

# Direct chat
curl -X POST "http://localhost:8000/api/v1/agui/tipidter/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"message": "Halo, saya butuh bantuan tentang kasus pembalakan liar"}'
```

## 📚 References

- [Agno AG-UI Documentation](https://docs.agno.com/applications/ag-ui/introduction)
- [AG-UI Protocol Specification](https://github.com/ag-ui-protocol/ag-ui)
- [Dojo Frontend](https://github.com/ag-ui-protocol/ag-ui)
- [CopilotKit Integration](https://docs.copilotkit.ai/)

---

**Implementasi ini memberikan foundation yang solid untuk AG-UI integration yang modular, scalable, dan maintainable.** 🎯 