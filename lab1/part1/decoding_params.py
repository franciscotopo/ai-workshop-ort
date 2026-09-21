from typing import Dict, Any
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser


from config import HF_TOKEN, MODEL_ID
from .basics import prompt

consulta: str = "Escribe una analogía breve para explicar RAG a un público no técnico."

configuraciones: Dict[str, Dict[str, Any]] = {
    "baseline":   {"temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.0, "max_new_tokens": 128},
    "creativo":   {"temperature": 1.1, "top_p": 0.90, "repetition_penalty": 1.0, "max_new_tokens": 128},
    "controlado": {"temperature": 0.2, "top_p": 0.80, "repetition_penalty": 1.1, "max_new_tokens": 96},
}

for nombre, cfg in configuraciones.items():
    tmp_endpoint = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="conversational",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        repetition_penalty=cfg["repetition_penalty"],
        max_new_tokens=cfg["max_new_tokens"],
    )

    tmp_llm = ChatHuggingFace(llm=tmp_endpoint)
    salida = (prompt | tmp_llm | StrOutputParser()).invoke({"instruccion": consulta})
    
    print(f"\n==== {nombre} ({cfg}) ====\n{salida}")

# ========== top_p: ==========

# top_p (también llamado nucleus sampling) 
# en vez de elegir siempre entre todas las palabras posibles, el modelo arma una lista de las palabras más 
# probables hasta que juntan una probabilidad acumulada de top_p (por ej. 0.9 = 90%), y elige solo entre esas. 
# Un valor bajo (0.8) recorta más las opciones raras → texto más conservador. Un valor alto (0.95) deja más variedad disponible.

# ========== repetition_penalty: ==========

# repetition_penalty — penaliza al modelo por repetir palabras o frases que ya generó. 
# Un valor de 1.0 no penaliza nada (default). Valores mayores a 1.0 (por ej. 1.1) hacen que evite repetirse, 
# útil cuando el modelo tiende a quedar "trabado" repitiendo lo mismo.

# ========== max_new_tokens: ==========

# max_new_tokens — el límite máximo de tokens (fragmentos de palabras) que el modelo puede generar en la respuesta. 
# No afecta el estilo de la respuesta, solo su longitud máxima — si la respuesta natural sería más larga, se corta ahí.