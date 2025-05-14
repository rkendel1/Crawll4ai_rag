## 1. Estilo de Código Python y Pydantic AI

- Seguir PEP 8 en todo el código Python; usar `black` y `flake8` en CI.
- Emplear type hints en funciones, métodos y modelos Pydantic.
- Definir esquemas con Pydantic AI para inputs/outputs de agentes:
  ```python
  class TaskSchema(BaseModel):
      id: str
      description: str
  ```
- Cada agente hereda de `pydantic_ai.Agent`, con `validate_input` y `validate_output`.
- Incluir docstrings en formato Google o NumPy en clases, métodos y lógica compleja.

## 2. Estructura y Documentación

- Mantener `README.md` y README-task-master.md actualizados: instalación, despliegue y casos de uso.
- Cada microservicio FastAPI en src debe documentar rutas con docstrings y `response_model`.
- Usar el modelo unificado `APIResponse` y helper `api_response` para respuestas HTTP.
- Generar esquemas OpenAPI automáticos; revisar docs en entorno local.

## 3. Orquestación de Agentes (A2A y LangGraph)

- Definir capacidades y habilidades de cada agente en registry.py.
- Construir flujos con LangGraph en graph_builder.py.
- Mantener mensajes y esquemas de comunicación en messages.py y schemas.py.
- Versionar cambios de protocolo en `routing.py` y documentar en docs.

## 4. Integraciones: AWS Bedrock & Supabase

- Llamadas a Bedrock vía `boto3` en `providers/bedrock.py`; encapsular en un cliente reusable.
- Guardar vectores y metadatos en Supabase usando supabase.py y `vector_store.py`.
- Variables sensibles (API keys, secrets) únicamente en .env, docker-compose.yml o Vault.

## 5. Logging y Observabilidad

- Usar logger central en `src/core/logging.py` con helpers `log_info`, `log_error`, `log_span`.
- Instrumentar spans en orquestador y agentes para trazabilidad (OpenTelemetry opcional).
- No exponer datos sensibles en logs; filtrar PII y API keys.

## 6. Manejo de Errores y Retries

- Capturar excepciones específicas (`httpx.HTTPStatusError`, `botocore.exceptions.ClientError`).
- En endpoints FastAPI, usar `except Exception:  # noqa: E722` solo para fallback y logging.
- Implementar políticas de retry con backoff exponencial en llamadas a servicios externos.

## 7. Pruebas y Quality Assurance

- Pytest para unitarias e integración; cobertura mínima del 90% en core agents y orquestación.
- Testear modelos Pydantic, flujos LangGraph y endpoints FastAPI.
- Integrar pruebas E2E en CI con `test_prd_agent_e2e.py` y scripts de ejemplo en scripts.

## 8. Seguridad y Gestión de Credenciales

- Nunca hardcodear secretos; usar .env y AWS IAM roles.
- Validar TODO input con Pydantic. Rechazar datos no conformes antes de llamar al LLM.
- Forzar HTTPS en comunicaciones internas y externas. Configurar TLS en Kong (API Gateway).

## 9. Contenedores y Despliegue

- Dockerfile base en la raíz; separar build/runtime para optimización.
- docker-compose.yml orquesta backend, supabase y servicios auxiliares. Mantenerlo limpio.
- Scripts de arranque en start.sh y reset.sh para inicializar entornos dev y prod.
- Versionar imágenes con tags semánticos; usar `docker-compose -f docker-compose.yml -f docker-compose.s3.yml` para S3.

## 10. Control de Versiones y Pull Requests

- Seguir Git Flow: `feature/*`, `hotfix/*`, `release/*`.
- Mensajes de commit imperativos y descriptivos.
- Cada PR debe incluir descripción clara, referencias a tareas y resultados de tests.

## 11. Organización del Código

- Estructura principal:
  ```
  src/
    a2a/           # Protocolo y mensajes
    orchestration/ # Construcción y ejecución de grafos
    agents/        # Implementaciones de cada agente
    providers/     # Clientes AWS, Supabase, otros servicios
    core/          # Logging, configuración, utilidades
    api/           # FastAPI routes y modelos
  tests/           # Pytest
  ```
- Separar API, lógica de negocio, modelos y utilidades.

## 12. Mejores Prácticas Generales

- Priorizar dependencias maduras (FastAPI, Pydantic, boto3, supabase-py).
- Optimizar consultas y flujos para latencia mínima.
- Reutilizar patrones existentes vía herramientas RAG y graphBuilder.
- Documentar decisiones arquitectónicas críticas en `docs/ARCHITECTURE.md`.

## 13. Uso de RAG y Búsqueda de Conocimientos

- Antes de codificar, ejecutar siempre un query RAG contra la base vectorial de Supabase:
  - “¿Ejemplos de validación de input Pydantic AI en este proyecto?”
  - “¿Cómo orquestar flujo de tareas en LangGraph según prácticas actuales?”
- Si existe patrón: seguirlo. Si no, proponer convención y documentar en `docs/CONVENTIONS.md`.
- Registrar prompts y resultados en bedrock_detailed.log para trazabilidad.

## 13. Instalacion de dependencias
- Agregar dependencias en pyproject.toml

---

**Nota:** Mantener este documento actualizado. Cualquier mejora o nueva convención debe ser revisada y versionada.