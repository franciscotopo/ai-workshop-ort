from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from typing import List

from config import HF_TOKEN, MODEL_ID

### ========== Introducción ==========

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
# print(chain.invoke({"instruccion": "Explica en 3 frases qué es un LLM y nombra 2 casos de uso."}))

### ========== Ejercicio 1.1 ==========

# - Probar 3 variaciones de temperature y observar el cambio en estilo.
# - Cambiar el rol del system para forzar un estilo (p.ej., “responde con viñetas y máximo 3 líneas”).

instruccion: str = "Resume la diferencia entre entrenamiento y fine-tuning en 3 puntos."
temperaturas: List[float] = [0.0, 0.7, 1.2]

def ejecutar_variacion(temperature: float) -> str:

    # Crear endpoint conversacional con 'temperature' y devolver el texto

    endpoint = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="conversational",
        temperature=temperature,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=256,
    )

    llm_ej1 = ChatHuggingFace(llm=endpoint)    

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente de inteligencia artificial, conciso y claro"),
        ("human", f"Por favor: {instruccion}")
    ])

    chain = prompt_template | llm_ej1 | StrOutputParser()
    chain_output = chain.invoke({})
    
    return chain_output

# for t in temperaturas:
#     print(f"\n==== temperature: {t} ====")
#     print(ejecutar_variacion(t))

def ejecutar_estilo(instruccion: str, estilo: str) -> str:
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", estilo),
        ("human", instruccion)
    ])

    chain = prompt_template | llm | StrOutputParser()
    chain_output = chain.invoke({})

    return chain_output

estilos: List[str] = [
    "responde con vinietas y ejemplos, máximo 3 líneas",
    "responde con estilo formal, como si fueras un profesor de universidad",
    "responde como si la explicación fuera para un niño"
]

for e in estilos: 
    print(f"\n==== estilo forzado: {e} ====")
    print(ejecutar_estilo(instruccion, e))