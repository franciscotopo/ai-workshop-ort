# 📖 Contenidos:

- Setup y configuración de Hugging Face Inference API
- Uso básico de LLMs con LangChain (LCEL)
- Parámetros de decodificación y control de estilo
- Prompt engineering avanzado: zero-shot, few-shot, Chain of Thought, Role Prompting y salida estructurada (JSON)

## Al finalizar, podrás:

- Conectarte a un LLM instruct vía Hugging Face Inference API desde LangChain.
- Construir cadenas simples con `prompt | llm | parser`.
- Diseñar prompts efectivos y controlar formato de salida (incluido JSON).

# Lab 1 - Setup

## Pasos para reproducir el entorno

1. Crear entorno virtual: `python3 -m venv venv`
2. Activarlo: `source venv/bin/activate`
3. Instalar dependencias: `pip install -U "langchain==0.3.7" ...`
4. Guardar versiones exactas: `pip freeze > requirements.txt`

## Configuración del token de Hugging Face

1. Crear el token en https://huggingface.co/settings/tokens
2. Crear un archivo `.env` en /lab-1 con: `HF_TOKEN=hf_token`
3. En este laboratorio utilizamos el modelo Qwen/Qwen3-4B-Instruct-2507

## Cómo correr cada parte

Los scripts están organizados en carpetas por parte (`part-1/`, `part-2/`, etc.) e importan configuración compartida desde `config.py` en la raíz de `lab-1`. Por eso hay que correrlos como módulo, parado en `lab-1/`: `python -m part-1.basics`