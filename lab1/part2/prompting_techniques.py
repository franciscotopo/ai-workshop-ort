from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import MODEL_ID, HF_TOKEN

# ========== 1. Zero-shot vs Few-shot ==========

det_endpoint = HuggingFaceEndpoint(
    repo_id=MODEL_ID, 
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN, 
    temperature=0.0,                            # LLM más "determinista"
    max_new_tokens=128
)

llm_det = ChatHuggingFace(llm=det_endpoint)
parser = StrOutputParser()

# Patrón no obvio: acróstico O-V-E (Overfitting), cada línea empieza con esa letra
instruccion = (
    "Escribe EXACTAMENTE 3 líneas sobre 'overfitting'. "
    "Cada línea debe comenzar con O, luego V, luego E (en ese orden). "
    "6-10 palabras por línea. Sin texto extra."
)

zero_shot = ChatPromptTemplate.from_messages([
    ("system", "Sigue estrictamente las instrucciones del usuario."),
    ("human", "{instruccion}")
])

few_shot = ChatPromptTemplate.from_messages([
    ("system", "Sigue estrictamente las instrucciones del usuario."),
    # Ejemplo: el patrón se demuestra con otro tema y otro acróstico (R-E-G)
    ("human", "Escribe EXACTAMENTE 3 líneas sobre 'regularización'. "
              "Cada línea debe comenzar con R, luego E, luego G. "
              "6-10 palabras por línea. Sin texto extra."),
    ("ai", "R Reduce complejidad para evitar ajustes al ruido\n"
           "E Estabiliza el aprendizaje con penalizaciones adecuadas\n"
           "G Generaliza mejor limitando pesos excesivamente grandes"),
    # Ahora se pide el caso real con el acróstico O-V-E
    ("human", "{instruccion}")
])

print("=== ZERO-SHOT ===")
print((zero_shot | llm_det | parser).invoke({"instruccion": instruccion}))

print("\n=== FEW-SHOT ===")
print((few_shot | llm_det | parser).invoke({"instruccion": instruccion}))

# ========== 2. Role Prompting ==========

role_prompt = ChatPromptTemplate.from_messages([
    ("system", "Actúa como profesor de IA de nivel intermedio. Sé claro, estructurado y usa ejemplos sencillos."),
    ("human", "Explica brevemente qué es el aprendizaje por refuerzo y menciona 2 ejemplos de aplicación."),
])

print((role_prompt | llm_det | StrOutputParser()).invoke({}))

# ========== 3. Chain Of Thought ==========

problema = (
    "En una población, 1% tiene la enfermedad. La prueba tiene 90% de sensibilidad y 90% de especificidad. "
    "Si una persona da positivo, ¿cuál es la probabilidad (en %) de que realmente esté enferma?"
)

# SIN CoT: SOLO porcentaje en una línea (recorta tokens)

endpoint_sin_cot = HuggingFaceEndpoint(
    repo_id=MODEL_ID, 
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN, 
    temperature=0.0, 
    max_new_tokens=6
)

llm_sin_cot = ChatHuggingFace(llm=endpoint_sin_cot)

prompt_sin_cot = ChatPromptTemplate.from_messages([
    ("system", "Devuelve SOLO un número en formato porcentaje (ej: 8.33%). "
               "Sin explicaciones, sin ecuaciones, sin texto extra."),
    ("human", "{q}")
])

# CON CoT: piensa paso a paso y cierra con una línea final

endpoint_con_cot = HuggingFaceEndpoint(
    repo_id=MODEL_ID, 
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN, 
    temperature=0.0, 
    max_new_tokens=256
)

llm_con_cot = ChatHuggingFace(llm=endpoint_con_cot)

prompt_con_cot = ChatPromptTemplate.from_messages([
    ("system", "Tómate tu tiempo y piensa paso a paso usando Bayes. "
               "Al final, da una sola línea con: 'Respuesta final: <n>%'"),
    ("human", "{q}")
])

print("=== SIN CoT ===")
print((prompt_sin_cot | llm_sin_cot | parser).invoke({"q": problema}))

print("\n=== CON CoT ===")
print((prompt_con_cot | llm_con_cot | parser).invoke({"q": problema}))

# ========== 4. Salida Estructurada ==========

import json

json_prompt = ChatPromptTemplate.from_template(
    """
    Eres un asistente que devuelve SIEMPRE JSON válido. Dado un tema, devuelve un objeto con:
    - "titulo": string
    - "puntos_clave": lista de 3 strings
    - "dificultad": uno de ["básico", "intermedio", "avanzado"]
    Responde SOLO con JSON válido sin texto adicional.
    Tema: {tema}
    """
)

json_text = (json_prompt | llm_det | parser).invoke({"tema": "RAG"})
print(json_text)

data = json.loads(json_text)
assert set(["titulo", "puntos_clave", "dificultad"]).issubset(data.keys())
print("JSON válido con las claves requeridas.")