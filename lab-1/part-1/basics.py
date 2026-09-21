from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import HF_TOKEN, MODEL_ID

hf_endpoint = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.7,
    max_new_tokens=256,
)

llm = ChatHuggingFace(llm=hf_endpoint)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil y conciso."),
    ("human", "Responde a la siguiente instrucción: {instruccion}"),
])

chain = prompt | llm | StrOutputParser()
print(chain.invoke({"instruccion": "Explica en 3 frases qué es un LLM y nombra 2 casos de uso."}))

### Ejercicio 1.1

# - Probar 3 variaciones de temperature y observar el cambio en estilo.
# - Cambiar el rol del system para forzar un estilo (p.ej., “responde con viñetas y máximo 3 líneas”).
# - Pregunta sugerida: “Resume la diferencia entre entrenamiento y fine-tuning en 3 puntos.”

from typing import List

instruccion: str = "Resume la diferencia entre entrenamiento y fine-tuning en 3 puntos."
temperaturas: List[float] = [0.0, 0.7, 1.2]

def ejecutar_variacion(temperature: float) -> str:
    # TODO: crear endpoint conversacional con 'temperature' y devolver el texto
    # Debe usar: MODEL_ID, HF_TOKEN, task="conversational"
    raise NotImplementedError

for t in temperaturas:
    print(f"\n==== temperature: {t} ====")
    print(ejecutar_variacion(t))

def ejecutar_estilo(instruccion: str) -> str:
    # TODO: crear prompt de estilo (system) + LLM base conversacional y devolver el texto
    raise NotImplementedError

print("\n==== estilo forzado ====")
print(ejecutar_estilo(instruccion))