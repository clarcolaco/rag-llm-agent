rag-agent/
├── app/
│   ├── main.py                  # FastAPI + Rota Swagger
│   ├── rag_pipeline.py          # Pipeline principal RAG
│   ├── embedding.py             # Função de embedding
│   ├── vector_store.py          # Inserção e busca no Qdrant
│   ├── llm_wrapper.py           # LLM local (GPT4All, etc.)
│   ├── models.py                # Pydantic input/output
│   └── seed.py                  # Carrega 1000 frases iniciais
├── data/
│   └── frases.txt               # Contém 1000 frases (1 por linha)
├── docker-compose.yml          # Qdrant local
├── requirements.txt
└── README.md



## RAG Agent Organizacional

### Como rodar
```bash
git clone ...
cd rag-agent
pip install -r requirements.txt
docker-compose up -d
python app/seed.py
uvicorn app.main:app --reload
```

### Endpoints
- POST `/query` – Envie uma pergunta, retorna resposta baseada nas frases organizacionais
- POST `/frase` – Adiciona nova frase

### Swagger UI
Acesse em: http://localhost:8000/docs
"""
