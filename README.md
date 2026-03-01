# Multilingual Medical Intake — Full-Stack Production Build Guide

**Everything you need to go from zero to a live, deployed project.**

Stack: Pinch SDK · LangGraph · LangChain · Claude API · FastAPI · React · PostgreSQL · Redis · Docker · Railway / Render

---

## Table of Contents

1. [Repo & Monorepo Setup](#1-repo--monorepo-setup)
2. [Environment & Secrets Management](#2-environment--secrets-management)
3. [Database Schema](#3-database-schema)
4. [Backend — FastAPI Server](#4-backend--fastapi-server)
5. [Pinch Audio Bridge](#5-pinch-audio-bridge)
6. [LangGraph Intake Graph](#6-langgraph-intake-graph)
7. [Claude Prompts](#7-claude-prompts)
8. [MCP Server (EHR Tools)](#8-mcp-server-ehr-tools)
9. [Frontend — React Kiosk](#9-frontend--react-kiosk)
10. [Auth & Session Security](#10-auth--session-security)
11. [Docker Setup](#11-docker-setup)
12. [Deployment (Railway)](#12-deployment-railway)
13. [Monitoring & Observability](#13-monitoring--observability)
14. [Testing Strategy](#14-testing-strategy)
15. [Go-Live Checklist](#15-go-live-checklist)

---

## 1. Repo & Monorepo Setup

Use a monorepo. It keeps WebSocket contracts, shared types, and env management in one place.

```
medical-intake/
├── backend/               # FastAPI Python app
│   ├── app/
│   │   ├── main.py
│   │   ├── ws/            # WebSocket handlers
│   │   ├── graph/         # LangGraph nodes + state
│   │   ├── pinch/         # Pinch bridge
│   │   ├── mcp/           # MCP server
│   │   ├── tools/         # LangChain tools
│   │   ├── db/            # SQLAlchemy models + migrations
│   │   └── config.py      # Settings (pydantic-settings)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── alembic/           # DB migrations
├── frontend/              # React app
│   ├── src/
│   │   ├── views/
│   │   │   ├── PatientKiosk.jsx
│   │   │   └── ProviderDashboard.jsx
│   │   ├── hooks/
│   │   │   ├── useAudioCapture.js
│   │   │   ├── usePinchPlayback.js
│   │   │   └── useIntakeSession.js
│   │   ├── components/
│   │   └── App.jsx
│   ├── Dockerfile
│   └── package.json
├── mcp-server/            # Optional standalone MCP process
│   ├── server.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
├── docker-compose.prod.yml
└── .env.example
```

Initialize:

```bash
git init medical-intake
cd medical-intake
mkdir -p backend/app/{ws,graph,pinch,mcp,tools,db} frontend/src/{views,hooks,components} mcp-server
```

---

## 2. Environment & Secrets Management

### `.env.example` (commit this, never `.env`)

```env
# Pinch
PINCH_API_KEY=

# Anthropic
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-opus-4-6

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/medical_intake
REDIS_URL=redis://localhost:6379

# App
SECRET_KEY=                    # openssl rand -hex 32
ALLOWED_ORIGINS=http://localhost:5173,https://yourdomain.com
ENVIRONMENT=development        # development | production

# MCP
EHR_MCP_SERVER_URL=http://localhost:8001

# Optional: Epic / FHIR EHR integration
FHIR_BASE_URL=
FHIR_CLIENT_ID=
FHIR_CLIENT_SECRET=
```

### `backend/app/config.py`

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    pinch_api_key: str
    anthropic_api_key: str
    anthropic_model: str = "claude-opus-4-6"
    database_url: str
    redis_url: str
    secret_key: str
    allowed_origins: list[str] = ["http://localhost:5173"]
    environment: str = "development"
    ehr_mcp_server_url: str = "http://localhost:8001"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

## 3. Database Schema

### `backend/app/db/models.py`

```python
from sqlalchemy import Column, String, Text, JSON, DateTime, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
import uuid
import enum
from datetime import datetime, timezone

class Base(AsyncAttrs, DeclarativeBase):
    pass

class TriageLevel(str, enum.Enum):
    routine = "routine"
    urgent = "urgent"
    emergency = "emergency"

class IntakeSession(Base):
    __tablename__ = "intake_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_language = Column(String(10), nullable=False)       # "es-ES"
    provider_language = Column(String(10), default="en-US")
    status = Column(String(20), default="active")               # active | complete | abandoned
    triage_level = Column(Enum(TriageLevel), nullable=True)
    transcript = Column(JSON, default=list)                     # [{role, content, ts}]
    chief_complaint = Column(Text, nullable=True)
    clinical_summary = Column(JSON, nullable=True)              # structured EHR JSON
    ehr_written = Column(String(1), default="N")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
```

### Migrations with Alembic

```bash
cd backend
pip install alembic
alembic init alembic
# Edit alembic/env.py to import your Base and use DATABASE_URL
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

---

## 4. Backend — FastAPI Server

### `backend/app/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import get_settings
from app.db.session import engine, Base
from app.ws.intake import router as intake_router
from app.ws.provider import router as provider_router

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(title="Medical Intake API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(intake_router, prefix="/ws")
app.include_router(provider_router, prefix="/ws")

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### `backend/app/db/session.py`

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False, pool_size=10)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

---

## 5. Pinch Audio Bridge

This is the core. One WebSocket per patient session. Audio in, transcript + translated audio out.

### `backend/app/pinch/bridge.py`

```python
import asyncio
from pinch import PinchClient
from pinch.session import SessionParams
from app.config import get_settings

settings = get_settings()

class PinchBridge:
    """
    Manages a single Pinch streaming session.
    Decouples audio I/O from LangGraph via asyncio queues.
    """

    def __init__(self, source_language: str, target_language: str):
        self.source_language = source_language
        self.target_language = target_language
        self.client = PinchClient(api_key=settings.pinch_api_key)
        self.stream = None
        self.session = None

        # Queues for decoupled async communication
        self.final_transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self.audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.partial_transcript_queue: asyncio.Queue[str] = asyncio.Queue()

        self._event_task: asyncio.Task | None = None
        self._closed = False

    async def start(self):
        self.session = self.client.create_session(
            SessionParams(
                source_language=self.source_language,
                target_language=self.target_language,
            )
        )
        self.stream = await self.client.connect_stream(
            self.session,
            audio_output_enabled=True,
        )
        # Start event listener in background
        self._event_task = asyncio.create_task(self._listen_events())

    async def _listen_events(self):
        async for event in self.stream.events():
            if event.type == "transcript":
                if event.is_final:
                    await self.final_transcript_queue.put(event.text)
                else:
                    await self.partial_transcript_queue.put(event.text)
            elif event.type == "audio":
                await self.audio_output_queue.put(event.pcm16_bytes)

    async def send_audio(self, pcm_bytes: bytes):
        """Called by WebSocket handler with each audio chunk from browser."""
        if not self._closed:
            await self.stream.send_pcm16(pcm_bytes, sample_rate=16000, channels=1)

    async def close(self):
        self._closed = True
        if self.stream:
            await self.stream.aclose()
        if self._event_task:
            self._event_task.cancel()
```

### `backend/app/ws/intake.py` — Patient WebSocket

```python
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.db.models import IntakeSession
from app.pinch.bridge import PinchBridge
from app.graph.runner import IntakeGraphRunner

router = APIRouter()

@router.websocket("/intake/{session_id}")
async def patient_intake_ws(
    websocket: WebSocket,
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    await websocket.accept()

    # Load or create session from DB
    session = await db.get(IntakeSession, session_id)
    if not session:
        await websocket.close(code=4004)
        return

    # Init Pinch bridge
    bridge = PinchBridge(
        source_language=session.patient_language,
        target_language=session.provider_language,
    )
    await bridge.start()

    # Init LangGraph runner
    graph_runner = IntakeGraphRunner(session_id=session_id, db=db)

    # Three concurrent tasks:
    # 1. Receive audio from browser → forward to Pinch
    # 2. Forward translated audio from Pinch → browser
    # 3. Forward final transcripts → LangGraph → get next question → Pinch for reverse translation

    async def receive_audio():
        try:
            while True:
                data = await websocket.receive_bytes()
                await bridge.send_audio(data)
        except WebSocketDisconnect:
            pass

    async def forward_translated_audio():
        while True:
            pcm = await bridge.audio_output_queue.get()
            try:
                await websocket.send_bytes(pcm)
            except Exception:
                break

    async def forward_partial_transcripts():
        while True:
            text = await bridge.partial_transcript_queue.get()
            try:
                await websocket.send_text(json.dumps({
                    "type": "partial_transcript",
                    "text": text
                }))
            except Exception:
                break

    async def process_final_transcripts():
        while True:
            text = await bridge.final_transcript_queue.get()
            # Send to provider dashboard via Redis pub/sub
            await publish_to_provider(session_id, text)
            # Run next LangGraph node
            next_question = await graph_runner.process_transcript(text)
            if next_question:
                # Send text back through Pinch for reverse translation (EN → patient language)
                # In practice: create a separate Pinch session or use TTS endpoint
                await websocket.send_text(json.dumps({
                    "type": "agent_question",
                    "text": next_question
                }))

    try:
        await asyncio.gather(
            receive_audio(),
            forward_translated_audio(),
            forward_partial_transcripts(),
            process_final_transcripts(),
        )
    finally:
        await bridge.close()
        await graph_runner.save_state()
```

> **Note on reverse translation:** When the agent asks a question back to the patient, you need EN→ES translation. Create a second Pinch session with flipped `source_language`/`target_language` for the agent voice, or use Pinch's file-based `translate_file()` on short text-to-speech segments.

---

## 6. LangGraph Intake Graph

### `backend/app/graph/state.py`

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class IntakeState(TypedDict):
    session_id: str
    patient_language: str
    transcript_history: Annotated[list, add_messages]
    chief_complaint: str
    symptoms: list[str]
    triage_level: str          # "routine" | "urgent" | "emergency"
    medical_history: dict
    current_medications: list[str]
    allergies: str
    clinical_summary: dict
    ehr_written: bool
    graph_phase: str           # tracks current node name
    next_question: str         # question to speak back to patient
```

### `backend/app/graph/intake_graph.py`

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.graph.state import IntakeState
from app.graph import nodes
from app.config import get_settings

settings = get_settings()

def build_intake_graph():
    builder = StateGraph(IntakeState)

    builder.add_node("greeting",          nodes.greeting_node)
    builder.add_node("chief_complaint",   nodes.chief_complaint_node)
    builder.add_node("symptom_detail",    nodes.symptom_detail_node)
    builder.add_node("triage_check",      nodes.triage_check_node)
    builder.add_node("urgent_protocol",   nodes.urgent_protocol_node)
    builder.add_node("history_meds",      nodes.history_meds_node)
    builder.add_node("generate_summary",  nodes.generate_summary_node)
    builder.add_node("ehr_write",         nodes.ehr_write_node)

    builder.add_edge(START, "greeting")
    builder.add_edge("greeting", "chief_complaint")
    builder.add_edge("chief_complaint", "symptom_detail")
    builder.add_edge("symptom_detail", "triage_check")

    # Conditional routing on triage result
    builder.add_conditional_edges(
        "triage_check",
        lambda state: state["triage_level"],
        {
            "routine":   "history_meds",
            "urgent":    "urgent_protocol",
            "emergency": "urgent_protocol",
        }
    )

    builder.add_edge("urgent_protocol",  "generate_summary")
    builder.add_edge("history_meds",     "generate_summary")
    builder.add_edge("generate_summary", "ehr_write")
    builder.add_edge("ehr_write",        END)

    # Use Postgres checkpointer for persistent state across WS reconnects
    checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url)
    return builder.compile(checkpointer=checkpointer)

intake_graph = build_intake_graph()
```

### `backend/app/graph/nodes.py`

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from app.graph.state import IntakeState
from app.graph import prompts
from app.config import get_settings

settings = get_settings()
llm = ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key)
llm_json = llm.bind(response_format={"type": "json_object"})


def greeting_node(state: IntakeState) -> dict:
    return {
        "graph_phase": "greeting",
        "next_question": "Hello! I'm here to help collect some information before your appointment. Can you tell me today's date of birth to confirm your identity?",
    }


def chief_complaint_node(state: IntakeState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.CHIEF_COMPLAINT_SYSTEM),
        ("human", "Transcript so far:\n{history}\n\nExtract the chief complaint and generate the next follow-up question. Respond in JSON: {{\"chief_complaint\": str, \"next_question\": str}}")
    ])
    chain = prompt | llm_json | JsonOutputParser()
    result = chain.invoke({"history": _format_history(state["transcript_history"])})
    return {
        "chief_complaint": result["chief_complaint"],
        "next_question": result["next_question"],
        "graph_phase": "chief_complaint",
    }


def symptom_detail_node(state: IntakeState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.SYMPTOM_DETAIL_SYSTEM),
        ("human", "Chief complaint: {complaint}\n\nTranscript:\n{history}\n\nRespond JSON: {{\"symptoms\": [str], \"next_question\": str}}")
    ])
    chain = prompt | llm_json | JsonOutputParser()
    result = chain.invoke({
        "complaint": state["chief_complaint"],
        "history": _format_history(state["transcript_history"]),
    })
    return {
        "symptoms": result["symptoms"],
        "next_question": result["next_question"],
        "graph_phase": "symptom_detail",
    }


def triage_check_node(state: IntakeState) -> dict:
    """
    Critical node. Must reliably return triage_level.
    Uses strict JSON mode.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.TRIAGE_SYSTEM),
        ("human", "Symptoms: {symptoms}\nChief complaint: {complaint}\n\nRespond ONLY JSON: {{\"triage_level\": \"routine\" | \"urgent\" | \"emergency\", \"reason\": str}}")
    ])
    chain = prompt | llm_json | JsonOutputParser()
    result = chain.invoke({
        "symptoms": ", ".join(state["symptoms"]),
        "complaint": state["chief_complaint"],
    })
    return {
        "triage_level": result["triage_level"],
        "graph_phase": "triage_check",
        "next_question": None,
    }


def urgent_protocol_node(state: IntakeState) -> dict:
    return {
        "graph_phase": "urgent_protocol",
        "next_question": "I'm flagging this as urgent. A provider will be with you shortly. Can you confirm your current pain level on a scale of 1 to 10?",
    }


def history_meds_node(state: IntakeState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.HISTORY_MEDS_SYSTEM),
        ("human", "Transcript:\n{history}\n\nRespond JSON: {{\"medical_history\": dict, \"current_medications\": [str], \"allergies\": str, \"next_question\": str}}")
    ])
    chain = prompt | llm_json | JsonOutputParser()
    result = chain.invoke({"history": _format_history(state["transcript_history"])})
    return {
        "medical_history": result["medical_history"],
        "current_medications": result["current_medications"],
        "allergies": result["allergies"],
        "next_question": result["next_question"],
        "graph_phase": "history_meds",
    }


def generate_summary_node(state: IntakeState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.CLINICAL_SUMMARY_SYSTEM),
        ("human", """
Session data:
- Chief complaint: {complaint}
- Symptoms: {symptoms}
- Triage level: {triage}
- Medical history: {history}
- Current medications: {meds}
- Allergies: {allergies}
- Full transcript: {transcript}

Generate a complete clinical intake summary as JSON matching this schema:
{{
  "chief_complaint": str,
  "hpi": str,
  "symptoms": [str],
  "triage_level": str,
  "red_flags": [str],
  "relevant_history": str,
  "allergies": str,
  "current_medications": [str],
  "provider_note": str,
  "recommended_action": str
}}
""")
    ])
    chain = prompt | llm_json | JsonOutputParser()
    result = chain.invoke({
        "complaint": state["chief_complaint"],
        "symptoms": ", ".join(state["symptoms"]),
        "triage": state["triage_level"],
        "history": state["medical_history"],
        "meds": ", ".join(state["current_medications"]),
        "allergies": state["allergies"],
        "transcript": _format_history(state["transcript_history"]),
    })
    return {
        "clinical_summary": result,
        "graph_phase": "generate_summary",
        "next_question": "Thank you. Your intake is complete. A member of our team will be with you shortly.",
    }


def ehr_write_node(state: IntakeState) -> dict:
    # Call MCP tool or direct DB write
    from app.tools.ehr_tools import write_intake_summary
    write_intake_summary.invoke({
        "session_id": state["session_id"],
        "summary": state["clinical_summary"],
        "triage_level": state["triage_level"],
    })
    return {"ehr_written": True, "graph_phase": "complete"}


def _format_history(history: list) -> str:
    return "\n".join(f"[{m['role'].upper()}] {m['content']}" for m in history)
```

### `backend/app/graph/runner.py`

```python
from app.graph.intake_graph import intake_graph
from app.graph.state import IntakeState
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

class IntakeGraphRunner:
    def __init__(self, session_id: str, db: AsyncSession):
        self.session_id = session_id
        self.db = db
        self.config = {"configurable": {"thread_id": session_id}}

    async def process_transcript(self, patient_text: str) -> str | None:
        """
        Receives a final transcript from Pinch,
        appends to state, advances the graph,
        returns the next question string.
        """
        current = await intake_graph.aget_state(self.config)
        values = current.values if current.values else self._initial_state()

        # Append patient turn to history
        values["transcript_history"].append({
            "role": "patient",
            "content": patient_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Advance graph by one node
        result = await intake_graph.ainvoke(values, self.config)

        # Append agent turn to history
        if result.get("next_question"):
            result["transcript_history"].append({
                "role": "agent",
                "content": result["next_question"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        return result.get("next_question")

    def _initial_state(self) -> IntakeState:
        return IntakeState(
            session_id=self.session_id,
            patient_language="es-ES",
            transcript_history=[],
            chief_complaint="",
            symptoms=[],
            triage_level="",
            medical_history={},
            current_medications=[],
            allergies="",
            clinical_summary={},
            ehr_written=False,
            graph_phase="start",
            next_question="",
        )

    async def save_state(self):
        state = await intake_graph.aget_state(self.config)
        if state and state.values:
            from app.db.models import IntakeSession
            session = await self.db.get(IntakeSession, self.session_id)
            if session:
                session.transcript = state.values.get("transcript_history", [])
                session.triage_level = state.values.get("triage_level")
                session.clinical_summary = state.values.get("clinical_summary")
                session.chief_complaint = state.values.get("chief_complaint")
                session.status = "complete" if state.values.get("ehr_written") else "active"
                session.completed_at = datetime.now(timezone.utc) if state.values.get("ehr_written") else None
                await self.db.commit()
```

---

## 7. Claude Prompts

### `backend/app/graph/prompts.py`

```python
CHIEF_COMPLAINT_SYSTEM = """
You are a clinical intake assistant. Your job is to identify the patient's chief complaint 
from their statements and ask a focused follow-up question.

Rules:
- Be warm, calm, and clear.
- Never diagnose. Only collect.
- Keep questions to one at a time.
- Output only valid JSON. No prose outside the JSON object.
"""

SYMPTOM_DETAIL_SYSTEM = """
You are a clinical intake assistant collecting symptom details.
Extract all mentioned symptoms (onset, duration, severity, character, location, radiation, 
associated symptoms). Ask one clarifying follow-up question.
Output only valid JSON.
"""

TRIAGE_SYSTEM = """
You are a clinical triage classifier. Based on symptoms and chief complaint, classify:
- emergency: life-threatening (chest pain + radiation, stroke signs, anaphylaxis, severe bleeding)
- urgent: needs prompt attention within 2 hours (high fever, moderate pain, breathing difficulty)
- routine: can wait for standard appointment slot

Be conservative. When uncertain between urgent and emergency, choose emergency.
Output ONLY valid JSON with triage_level and reason. No other text.
"""

HISTORY_MEDS_SYSTEM = """
You are a clinical intake assistant collecting medical history.
Extract: past diagnoses, surgical history, current medications (name + dose if mentioned), 
allergies (drug and environmental). Ask one follow-up question to complete the picture.
Output only valid JSON.
"""

CLINICAL_SUMMARY_SYSTEM = """
You are a clinical documentation specialist. Generate a complete, accurate intake note 
in standard clinical language suitable for a physician to review before seeing the patient.

The HPI (History of Present Illness) should be written in prose using OLDCARTS format 
(Onset, Location, Duration, Character, Aggravating factors, Relieving factors, Timing, Severity).

provider_note should be a brief, direct clinical summary as if written by an experienced triage nurse.
recommended_action should be one clear sentence.

Output only valid JSON.
"""
```

---

## 8. MCP Server (EHR Tools)

This runs as a separate service. Claude connects to it via the MCP protocol to write summaries and look up patient history.

### `mcp-server/server.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
import json

app = FastAPI(title="EHR MCP Server")

# MCP tool registry
MCP_TOOLS = [
    {
        "name": "write_intake_summary",
        "description": "Write a completed intake summary to the EHR system",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "summary": {"type": "object"},
                "triage_level": {"type": "string", "enum": ["routine", "urgent", "emergency"]},
            },
            "required": ["session_id", "summary", "triage_level"],
        }
    },
    {
        "name": "get_patient_history",
        "description": "Retrieve existing patient records by MRN",
        "input_schema": {
            "type": "object",
            "properties": {
                "mrn": {"type": "string"},
            },
            "required": ["mrn"],
        }
    },
    {
        "name": "flag_triage",
        "description": "Notify the provider dashboard of triage level",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "level": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["session_id", "level"],
        }
    },
]

@app.get("/mcp/tools")
async def list_tools():
    return {"tools": MCP_TOOLS}

@app.post("/mcp/tools/{tool_name}")
async def call_tool(tool_name: str, body: dict):
    if tool_name == "write_intake_summary":
        # In production: write to FHIR endpoint or Postgres
        print(f"Writing summary for session {body['session_id']}")
        return {"success": True, "session_id": body["session_id"]}

    elif tool_name == "get_patient_history":
        # In production: query EHR by MRN
        return {"mrn": body["mrn"], "history": [], "medications": [], "allergies": []}

    elif tool_name == "flag_triage":
        # In production: push to provider dashboard via Redis pub/sub
        return {"flagged": True, "level": body["level"]}

    return {"error": "unknown tool"}
```

### Calling MCP from LangGraph (in `ehr_write_node`)

```python
import httpx
from app.config import get_settings

async def call_mcp_tool(tool_name: str, inputs: dict) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{settings.ehr_mcp_server_url}/mcp/tools/{tool_name}",
            json=inputs,
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()
```

---

## 9. Frontend — React Kiosk

### `frontend/src/hooks/useAudioCapture.js`

```javascript
import { useRef, useCallback } from "react";

// Captures mic audio, downsamples to 16kHz PCM16, sends via WebSocket
export function useAudioCapture(websocketRef) {
  const contextRef = useRef(null);
  const processorRef = useRef(null);

  const start = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const context = new AudioContext({ sampleRate: 16000 });
    contextRef.current = context;

    const source = context.createMediaStreamSource(stream);

    // ScriptProcessor (legacy but universally supported; swap to AudioWorklet for production)
    const processor = context.createScriptProcessor(3200, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      const float32 = e.inputBuffer.getChannelData(0);
      const pcm16 = floatToPCM16(float32);
      if (websocketRef.current?.readyState === WebSocket.OPEN) {
        websocketRef.current.send(pcm16.buffer);
      }
    };

    source.connect(processor);
    processor.connect(context.destination);
  }, [websocketRef]);

  const stop = useCallback(() => {
    processorRef.current?.disconnect();
    contextRef.current?.close();
  }, []);

  return { start, stop };
}

function floatToPCM16(float32Array) {
  const pcm16 = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const clamped = Math.max(-1, Math.min(1, float32Array[i]));
    pcm16[i] = clamped < 0 ? clamped * 32768 : clamped * 32767;
  }
  return pcm16;
}
```

### `frontend/src/hooks/usePinchPlayback.js`

```javascript
import { useRef, useCallback } from "react";

// Receives translated PCM16 audio bytes from WebSocket, plays them
export function usePinchPlayback() {
  const contextRef = useRef(null);
  const SAMPLE_RATE = 16000;

  const init = useCallback(() => {
    contextRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
  }, []);

  const playChunk = useCallback((arrayBuffer) => {
    const context = contextRef.current;
    if (!context) return;

    const int16 = new Int16Array(arrayBuffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    const audioBuffer = context.createBuffer(1, float32.length, SAMPLE_RATE);
    audioBuffer.copyToChannel(float32, 0);

    const source = context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(context.destination);
    source.start();
  }, []);

  return { init, playChunk };
}
```

### `frontend/src/hooks/useIntakeSession.js`

```javascript
import { useRef, useState, useCallback } from "react";
import { useAudioCapture } from "./useAudioCapture";
import { usePinchPlayback } from "./usePinchPlayback";

const WS_BASE = import.meta.env.VITE_WS_URL || "ws://localhost:8000";

export function useIntakeSession(sessionId) {
  const wsRef = useRef(null);
  const [status, setStatus] = useState("idle");          // idle | connecting | active | complete
  const [partialTranscript, setPartialTranscript] = useState("");
  const [agentQuestion, setAgentQuestion] = useState("");

  const { playChunk, init: initPlayback } = usePinchPlayback();
  const { start: startCapture, stop: stopCapture } = useAudioCapture(wsRef);

  const connect = useCallback(async () => {
    setStatus("connecting");
    initPlayback();

    const ws = new WebSocket(`${WS_BASE}/ws/intake/${sessionId}`);
    wsRef.current = ws;

    ws.binaryType = "arraybuffer";

    ws.onopen = async () => {
      setStatus("active");
      await startCapture();
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        // Translated audio from Pinch — play it
        playChunk(event.data);
      } else {
        const msg = JSON.parse(event.data);
        if (msg.type === "partial_transcript") {
          setPartialTranscript(msg.text);
        } else if (msg.type === "agent_question") {
          setAgentQuestion(msg.text);
          setPartialTranscript("");
        }
      }
    };

    ws.onclose = () => {
      stopCapture();
      setStatus("complete");
    };
  }, [sessionId]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
  }, []);

  return { status, partialTranscript, agentQuestion, connect, disconnect };
}
```

### `frontend/src/views/PatientKiosk.jsx`

```jsx
import { useIntakeSession } from "../hooks/useIntakeSession";

export default function PatientKiosk({ sessionId }) {
  const { status, partialTranscript, agentQuestion, connect, disconnect } = useIntakeSession(sessionId);

  return (
    <div style={{ padding: 40, fontFamily: "sans-serif", maxWidth: 700, margin: "0 auto" }}>
      <h1 style={{ fontSize: 32, marginBottom: 8 }}>Intake de bienvenida</h1>
      <p style={{ color: "#666", marginBottom: 32 }}>Hable cuando esté listo. Le escuchamos.</p>

      {status === "idle" && (
        <button onClick={connect} style={btnStyle}>
          Iniciar consulta
        </button>
      )}

      {status === "active" && (
        <>
          <div style={{ background: "#f0f4ff", borderRadius: 12, padding: 24, marginBottom: 20 }}>
            <p style={{ fontSize: 20, fontWeight: 600 }}>{agentQuestion}</p>
          </div>
          {partialTranscript && (
            <p style={{ color: "#888", fontStyle: "italic" }}>{partialTranscript}…</p>
          )}
          <button onClick={disconnect} style={{ ...btnStyle, background: "#e53e3e", marginTop: 24 }}>
            Finalizar
          </button>
        </>
      )}

      {status === "complete" && (
        <p style={{ fontSize: 20 }}>✅ Gracias. Su información ha sido registrada. Un médico le atenderá pronto.</p>
      )}
    </div>
  );
}

const btnStyle = {
  background: "#2563eb", color: "white", border: "none",
  borderRadius: 8, padding: "14px 28px", fontSize: 18, cursor: "pointer",
};
```

### `frontend/src/views/ProviderDashboard.jsx`

```jsx
import { useEffect, useRef, useState } from "react";

const WS_BASE = import.meta.env.VITE_WS_URL?.replace("ws://", "http://") || "http://localhost:8000";

export default function ProviderDashboard({ sessionId }) {
  const [transcript, setTranscript] = useState([]);
  const [summary, setSummary] = useState(null);
  const [triage, setTriage] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE.replace("http", "ws")}/ws/provider/${sessionId}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "transcript_line") {
        setTranscript(prev => [...prev, msg]);
      } else if (msg.type === "triage_flag") {
        setTriage(msg.level);
      } else if (msg.type === "summary_ready") {
        setSummary(msg.summary);
      }
    };

    return () => ws.close();
  }, [sessionId]);

  return (
    <div style={{ display: "flex", gap: 24, padding: 32, fontFamily: "sans-serif" }}>
      <div style={{ flex: 1 }}>
        <h2>Live Transcript (EN)</h2>
        {triage && (
          <div style={{ background: triage === "emergency" ? "#fee" : "#fef9c3", padding: 12, borderRadius: 8, marginBottom: 16 }}>
            ⚠️ Triage: <strong>{triage.toUpperCase()}</strong>
          </div>
        )}
        <div style={{ height: 500, overflowY: "auto", background: "#f8f8f8", borderRadius: 8, padding: 16 }}>
          {transcript.map((t, i) => (
            <p key={i} style={{ marginBottom: 8 }}>
              <strong style={{ color: t.role === "agent" ? "#2563eb" : "#16a34a" }}>
                [{t.role.toUpperCase()}]
              </strong>{" "}{t.text}
            </p>
          ))}
        </div>
      </div>

      {summary && (
        <div style={{ width: 380, background: "#f0fdf4", borderRadius: 12, padding: 24 }}>
          <h2>Clinical Summary</h2>
          <p><strong>Chief Complaint:</strong> {summary.chief_complaint}</p>
          <p><strong>HPI:</strong> {summary.hpi}</p>
          <p><strong>Triage:</strong> {summary.triage_level}</p>
          <p><strong>Allergies:</strong> {summary.allergies}</p>
          <p><strong>Medications:</strong> {summary.current_medications?.join(", ")}</p>
          <p><strong>Provider Note:</strong> {summary.provider_note}</p>
          <p><strong>Recommended Action:</strong> {summary.recommended_action}</p>
        </div>
      )}
    </div>
  );
}
```

---

## 10. Auth & Session Security

For a live medical product, auth is non-negotiable.

**Session creation flow:**
1. Provider creates an intake session via `POST /api/sessions` (protected with API key or JWT)
2. Server returns a `session_id` + short-lived `session_token` (signed JWT, 30 min TTL)
3. Kiosk URL includes token: `https://kiosk.yourapp.com/intake?token=<jwt>`
4. WebSocket connection validates token on `connect`

```python
# backend/app/ws/intake.py — add token validation
from fastapi import Query
import jwt

@router.websocket("/intake/{session_id}")
async def patient_intake_ws(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        if payload["session_id"] != session_id:
            await websocket.close(code=4003)
            return
    except jwt.ExpiredSignatureError:
        await websocket.close(code=4001)
        return
    except jwt.InvalidTokenError:
        await websocket.close(code=4003)
        return

    await websocket.accept()
    # ... rest of handler
```

**HIPAA considerations for production:**
- Use HTTPS/WSS exclusively (TLS termination at load balancer)
- Encrypt `clinical_summary` column at rest (use pgcrypto or column-level encryption)
- Audit log every access to patient records
- Set `DATABASE_URL` to a HIPAA-compliant Postgres host (AWS RDS with encryption, or Supabase with row-level security)
- Never log raw transcript content to stdout in production

---

## 11. Docker Setup

### `docker-compose.yml` (development)

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: intake
      POSTGRES_PASSWORD: intake
      POSTGRES_DB: medical_intake
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  backend:
    build: ./backend
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    volumes:
      - ./backend:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  mcp-server:
    build: ./mcp-server
    env_file: .env
    ports:
      - "8001:8001"
    command: uvicorn server:app --host 0.0.0.0 --port 8001 --reload

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    command: npm run dev -- --host

volumes:
  pgdata:
```

### `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `backend/requirements.txt`

```
fastapi>=0.115
uvicorn[standard]>=0.30
websockets>=12
pydantic-settings>=2
sqlalchemy[asyncio]>=2
asyncpg>=0.29
alembic>=1.13
pinch-sdk>=0.1.2
langchain>=0.3
langchain-anthropic>=0.3
langgraph>=0.2
langgraph-checkpoint-postgres>=0.1
anthropic>=0.40
redis>=5
httpx>=0.27
python-jose[cryptography]>=3.3
```

---

## 12. Deployment (Railway)

Railway is the fastest path to production for this stack. It handles Postgres, Redis, and multiple services natively.

**Step 1 — Push to GitHub**

```bash
git remote add origin https://github.com/yourname/medical-intake
git push -u origin main
```

**Step 2 — Create Railway project**

```bash
npm install -g @railway/cli
railway login
railway init
```

**Step 3 — Add services**

From Railway dashboard:
- Add PostgreSQL plugin → get `DATABASE_URL`
- Add Redis plugin → get `REDIS_URL`
- Add service from GitHub → point to `backend/` → set root dir
- Add service from GitHub → point to `mcp-server/` → set root dir
- Add service from GitHub → point to `frontend/` → set root dir

**Step 4 — Set env vars in Railway**

Copy all values from `.env` into Railway's environment variable panel per service.

**Step 5 — Set start commands**

Backend: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
MCP: `uvicorn server:app --host 0.0.0.0 --port $PORT`
Frontend: `npm run build && npx serve dist -l $PORT`

**Step 6 — Run migrations on deploy**

Add a deploy command in Railway for the backend service:
```
alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Step 7 — Custom domain**

In Railway: Settings → Domains → Add custom domain → point your DNS CNAME to Railway's provided endpoint. TLS is automatic.

**Alternative: Render**
Same flow but via `render.yaml`. Both Railway and Render support WebSocket connections natively — no special config needed.

---

## 13. Monitoring & Observability

**Logging**

```python
# backend/app/main.py — add structured logging
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        })

logging.getLogger().setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger().addHandler(handler)
```

**LangSmith tracing (LangGraph)**

```python
# Add to .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=medical-intake
```

All LangGraph node calls, token usage, and latency are automatically traced in LangSmith. Critical for debugging triage classification failures.

**Sentry**

```bash
pip install sentry-sdk[fastapi]
```

```python
import sentry_sdk
sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), environment=settings.environment)
```

**Health endpoint with dependency checks**

```python
@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok", "db": "ok"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "degraded", "db": str(e)})
```

---

## 14. Testing Strategy

### Unit test — triage node

```python
# tests/test_triage_node.py
import pytest
from app.graph.nodes import triage_check_node
from app.graph.state import IntakeState

def base_state(**kwargs) -> IntakeState:
    defaults = IntakeState(
        session_id="test-001",
        patient_language="es-ES",
        transcript_history=[],
        chief_complaint="",
        symptoms=[],
        triage_level="",
        medical_history={},
        current_medications=[],
        allergies="",
        clinical_summary={},
        ehr_written=False,
        graph_phase="",
        next_question="",
    )
    defaults.update(kwargs)
    return defaults

def test_emergency_triage():
    state = base_state(
        chief_complaint="Chest pain radiating to left arm",
        symptoms=["chest pain", "arm pain", "sweating", "nausea"],
    )
    result = triage_check_node(state)
    assert result["triage_level"] in ("urgent", "emergency")

def test_routine_triage():
    state = base_state(
        chief_complaint="Sore throat for 2 days",
        symptoms=["sore throat", "mild fever"],
    )
    result = triage_check_node(state)
    assert result["triage_level"] == "routine"
```

### Integration test — Pinch bridge (mock)

```python
# tests/test_pinch_bridge.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.pinch.bridge import PinchBridge

@pytest.mark.asyncio
async def test_bridge_queues_final_transcript():
    bridge = PinchBridge("es-ES", "en-US")

    mock_event = MagicMock()
    mock_event.type = "transcript"
    mock_event.is_final = True
    mock_event.text = "Me duele el pecho"

    async def fake_events():
        yield mock_event

    with patch.object(bridge, "stream") as mock_stream:
        mock_stream.events = fake_events
        await bridge._listen_events()

    result = await bridge.final_transcript_queue.get()
    assert result == "Me duele el pecho"
```

---

## 15. Go-Live Checklist

**Infrastructure**
- [ ] PostgreSQL with daily automated backups enabled
- [ ] Redis persistence enabled (AOF or RDB)
- [ ] TLS/WSS on all WebSocket connections
- [ ] Env vars set in Railway/Render (not in code)
- [ ] Health endpoint monitored by uptime service (BetterStack, UptimeRobot)

**Security**
- [ ] Session token expiry ≤ 60 min
- [ ] `clinical_summary` column encrypted at rest
- [ ] No patient data in application logs
- [ ] CORS restricted to production domain only
- [ ] Rate limiting on session creation endpoint (`slowapi`)

**Pinch**
- [ ] Confirmed API key is production key, not test key
- [ ] Verified supported language codes for all target languages (`es-ES`, `pt-BR`, etc.)
- [ ] Tested stream reconnect behavior on network drop
- [ ] `stream.aclose()` always called in `finally` block

**LangGraph**
- [ ] LangSmith tracing enabled in production
- [ ] Triage node tested against ≥20 symptom scenarios before launch
- [ ] Postgres checkpointer confirmed working (sessions survive backend restart)

**Frontend**
- [ ] `VITE_WS_URL` points to production WSS endpoint
- [ ] Tested on kiosk hardware (touch screen, specific browser)
- [ ] Audio autoplay policy handled (user gesture required before mic start)
- [ ] Tested on slow network (3G simulation in DevTools)

**Legal / Compliance**
- [ ] Privacy notice displayed to patient before session starts
- [ ] Business Associate Agreement (BAA) signed with all cloud providers if handling PHI
- [ ] Intake data retention policy defined and implemented
- [ ] Provider reviewed and approved the clinical summary template