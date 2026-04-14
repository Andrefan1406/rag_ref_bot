# RAG Ref Bot (MVP)

MVP RAG-система:
- /search (OpenAlex RU)
- /ingest (скачать PDF по ссылке и добавить в базу)
- /upload_pdf (загрузить PDF вручную)
- /chat (поиск по базе и выдача фрагментов)

## Установка
pip install -r requirements.txt

## Запуск
uvicorn app.main:app --reload --port 8000

Открыть:
http://127.0.0.1:8000/docs
