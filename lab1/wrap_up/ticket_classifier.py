# ========== Ejercicio Integrador: Clasificador de tickets de soporte ==========

# Objetivo: Desarrollar un script que integre un LLM para procesar tickets de soporte técnico, devolviendo un JSON.
# El equipo debe construir un clasificador que reciba tickets de soporte técnico (texto libre) y usando un LLM, 
# devuelva para cada uno un JSON con:

# json { 
#   "categoría": "facturación | técnico | cuenta | otro", 
#   "urgencia": "baja | media | alta", 
#   "resumen": "una oración que resuma el ticket" 
# }

# Se entrega un set de 5 tickets de ejemplo con su clasificación esperada (ver tabla debajo).

# Restricción clave: no basta con "que funcione una vez". El equipo debe demostrar que iteró el prompt para mejorar precisión y formato, y debe poder explicar por qué la versión final es mejor que la primera.

# ID |	Ticket	                                                                          | Categoría	 | Urgencia
# 1	 |  "No me llega el correo de confirmación de pago desde hace 2 días"	              | facturación	 | media
# 2	 |  "La app se cierra sola cada vez que intento subir una foto"	                      | técnico	     | alta
# 3	 |  "Quiero cambiar el email asociado a mi cuenta"	                                  | cuenta	     | baja
# 4	 |  "Me cobraron dos veces la suscripción de este mes, necesito el reembolso urgente" |	facturación	 | alta
# 5	 |  "¿Tienen alguna guía de uso para nuevos usuarios?"	                              | otro	     | baja

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import json

from part2.exercises import llm, parser

parser = StrOutputParser()

def ticket_classifier(claim: str) -> str:

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Actúa como asistente de clasificación de soporte técnico y peticiones de de usuarios de una app."
                   "Para cada petición, devuelve un único JSON con los campos categoría (facturación, técnico, cuenta, otro), urgencia (baja, media, alta), resumen ()"
                   "En caso de no tener clara una respuesta, realiza un razonamiento estructurado sobre por qué un ticket podría entrar en una clasificación u otra, y devuelve el resultado único en JSON luego."),
        ("user", "No me llega el correo de confirmación de pago desde hace 2 días"),
        ("ai", '{{"categoría": "facturación", "urgencia": "media", "resumen": "el usuario tiene un problema con la recepción de correos de confirmación de pagos." }}'),
        ("user", "La app se cierra sola cada vez que intento subir una foto"),
        ("ai", '{{"categoría": "técnico", "urgencia": "alta", "resumen": "el usuario reporta un bug en la app, que falla y se cierra al intentar subir una foto, lo cual degrada su experiencia de usuario." }}'),
        ("user", "Quiero cambiar el email asociado a mi cuenta"),
        ("ai", '{{"categoría": "cuenta", "urgencia": "baja", "resumen": "Pedido por parte del usuario para realizar un cambio de configuración en su cuenta." }}'),
        ("user", "Me cobraron dos veces la suscripción de este mes, necesito el reembolso urgente"),
        ("ai", '{{"categoría": "facturación", "urgencia": "alta", "resumen": "El usuario solicita un reembolso de sus suscripción mensual de inmediato tras un cobro duplicado" }}'),
        ("user", "¿Tienen alguna guía de uso para nuevos usuarios?"),
        ("ai", '{{"categoría": "otro", "urgencia": "baja", "resumen": "Solicitud de una guía de uso e instrucciones para nuevos usuarios" }}'),
        ("user", claim),
    ])

    output = (prompt | llm | parser).invoke({})

    return output

test_tickets = [
    "Hace una semana que no puedo iniciar sesión, me tira error de contraseña incorrecta",
    "El sistema me cobró en dólares en vez de en pesos, ¿pueden revisar?",
    "¿Cómo hago para exportar mis datos a un Excel?",
    "La app tarda muchísimo en cargar las fotos, casi no se puede usar",
]

for t in test_tickets:
    print(json.loads(ticket_classifier(t)))

# Resumen

# 1. Probé con una aproximación few-shot utilizando los ejemplos dados en la tabla.
# 2. Establecí un rol específico para que el modelo simule cierto comporatmiento para responder.
# 3. Agregué un Structred Output.
# 4. Agregué un CoT implícito en casos ambiguos.