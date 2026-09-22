# ========== Ejercicios ==========

# Resuelve los siguientes ejercicios. Modifica prompts y parámetros si es necesario y justifica 
# brevemente tus decisiones (en una celda de texto).

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from config import HF_TOKEN, MODEL_ID

hf_endpoint = HuggingFaceEndpoint(
    huggingfacehub_api_token=HF_TOKEN,
    repo_id=MODEL_ID,
    temperature=0.1,
    task="conversational",
    max_new_tokens=256,
)

llm = ChatHuggingFace(llm=hf_endpoint)
parser = StrOutputParser()

# ========== Ejercicio 2.1 — Zero-shot vs Few-shot ==========

# Tarea: explicar “regularización L2” en 3 viñetas claras para un público técnico.
# Paso 1 (zero-shot): crea un prompt sin ejemplos y observa el resultado.
# Paso 2 (few-shot): agrega 1-2 ejemplos de estilo y compara la salida.
# Pregunta guía: ¿mejoró la precisión o claridad con pocos ejemplos? Justifica.

def zero_shot_l2() -> str:
    # Construir prompt zero-shot (3 viñetas) y llamar al LLM
    tarea = "Explica el concepto de 'regularización L2' en 3 viñetas claras para un público técnico."

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responde la petición del usuario."),
        ("user", tarea)
    ])

    chain_output = (prompt | llm | parser).invoke({})
    
    return chain_output

    # === ZERO-SHOT ===
    # - **Restringe los coeficientes del modelo**: La regularización L2 (también conocida como Ridge) añade una penalización proporcional al cuadrado de los valores absolutos de los coeficientes del modelo. Esto impide que los coeficientes sean demasiado grandes, reduciendo el overfitting.
    # - **Promueve soluciones más simples y estables**: Al penalizar los grandes coeficientes, el modelo tiende a encontrar soluciones más simples y menos sensibles a ruido enlos datos. Esto mejora la estabilidad del modelo, especialmente cuando hay multicolinealidad o datos escasos.
    # - **Funciona como un método de selección de características indirecta**: Aunque no elimina directamente variables, los coeficientes reducidos por L2 tienden a acercarse a cero, lo que puede interpretarse como una forma de "reducción de características" y ayuda a interpretar el modelo de forma más clara.

def few_shot_l2() -> str:
    # Construir prompt con 1-2 ejemplos y llamar al LLM
    tarea = "Explica el concepto de 'regularización L2' en 3 viñetas claras para un público técnico."

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responde la petición del usuario."),
        ("user", "Explica el concepto de RAG en 3 viñetas claras para un público técnico"),
        ("ai", "- **Recupera información relevante**: Antes de generar la respuesta, busca en una base de datos externa los documentos más relacionados con la pregunta.\n"
               "- **Combina recuperación y generación**: El LLM usa esos documentos recuperados como contexto adicional para redactar una respuesta más precisa.\n"
               "- **Reduce alucinaciones**: Al basarse en información real y actualizada, RAG disminuye el riesgo de que el modelo invente datos."),
        ("user", tarea),
    ])

    chain_output = (prompt | llm | parser).invoke({})

    return chain_output

    # === FEW-SHOT ===
    # - **Añade un penalty a la magnitud de los pesos**: La regularización L2 (también llamada *ridge regularization*) suma al costo de entrenamiento el cuadrado de la norma de los pesos del modelo.
    # - **Previene overfitting al penalizar pesos grandes**: Al castigar los pesos muy grandes, el modelo se vuelve más simple y generalizable, evitando que se ajuste demasiado a los datos de entrenamiento.
    # - **Equilibra el rendimiento entre ajuste y generalización**: A diferencia de la regularización L1, L2 no hace que los pesos se ajusten a cero, sino que los reduce uniformemente, promoviendo una solución más estable y continua.

# print("\n=== ZERO-SHOT ===\n")
# print(zero_shot_l2())

# print("\n=== FEW-SHOT ===\n")
# print(few_shot_l2())

# Conclusión: este caso, el few-shot no mejoró demasiado porque el pedido del usuario (explicar un concepto técnico en 3 viñetas) 
# es un formato común y conocido. El modelo probablemente entiende cómo hacerlo bien sin ejemplos, sobre todo con temperature=0.1 
# (bastante determinista). Comparado esto con un ejemplo del acróstico O-V-E en prompting_techniques.py: ahí el few-shot sí importó, 
# porque el patrón pedido (cada línea empieza con una letra específica, en un orden dado) no es común y puede ser ambiguo 
# El modelo necesita ver un ejemplo concreto para entender qué se le está pidiendo, algo que la instrucción en texto sola no comunica bien.
# El few-shot ayuda más cuando el formato/patrón pedido es inusual o difícil de describir solo con palabras. 
# Cuando la tarea es estándar (como "3 viñetas claras"), el modelo la puede resolver bien en zero-shot, y agregar ejemplos no aporta demasiado.

# ========== Ejercicio 2.2 — Chain-of-Thought (CoT) ==========

# Tarea: dado un problema de evaluación de modelos, razonar paso a paso y entregar una conclusión final breve.
# Problema sugerido: “¿Por qué accuracy puede ser engañoso en un dataset muy desbalanceado y qué métrica alternativa usarías?”
# Pista: pide explícitamente “razona paso a paso y luego da una respuesta final breve en una línea”.

def cot_razonamiento(problema: str) -> str:
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Piensa paso a paso el problema. Deja registro de cada paso del razonamiento que das, justificando cada decisión para ir conectando las ideas."
                  "Al final del razonamiento, devuelve el resultado del problema en una sola línea"),
        ("user", problema)
    ])

    output = (prompt | llm | parser).invoke({})

    return output

# problema: str = "¿Por qué accuracy puede ser engañoso en un dataset muy desbalanceado y qué métrica alternativa usarías?"
# print("\n=== CHAIN-OF-THOUGHT ===\n")
# print(cot_razonamiento(problema))

# ========== Ejercicio 2.3 — Role prompting ==========

# Tarea: explicar el “sesgo de selección” a un equipo de data engineering con ejemplos concisos.
# Rol: “Actúa como líder técnico de datos; sé pragmático y directo, con viñetas concretas”.
# Objetivo: evaluar cómo cambia el estilo bajo un rol técnico específico.

def explicar_sesgo_seleccion() -> str:
    role = "Actúa como líder técnico de datos; sé pragmático y directo y da tus respuestas con viñetas concretas"
    request = "Explica el 'sesgo de selección' a un equipo de data engineering con ejemplos concisos"

    prompt = ChatPromptTemplate.from_messages([
        ("system", role),
        ("human", request),
    ])

    output = (prompt | llm | parser).invoke({})

    return output

print("\n=== ROLE PROMPTING ===\n")
print(explicar_sesgo_seleccion())


