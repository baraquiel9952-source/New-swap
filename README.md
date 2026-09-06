# Face Swap — proyecto unificado

Consolida en un solo proyecto lo que antes eran 5 proyectos separados en Vercel
(frontend, orquestador, detector, swapper, enhancer). Usa los mismos modelos
de código abierto que la mayoría de herramientas de este tipo:

| Etapa | Modelo | Peso |
|---|---|---|
| Detección + landmarks + identidad | InsightFace `buffalo_sc` (versión ligera) | **16 MB** |
| Intercambio de rostro | `inswapper_128_fp16.onnx` (versión ligera) | **~125 MB** |
| Restauración/nitidez | `GPEN-BFR-512.onnx` (versión ligera, ONNX puro) | **284 MB** |

**Total: ~425 MB** (antes ~1.2 GB) — y como GPEN corre solo con
`onnxruntime`, ya no hace falta instalar PyTorch/GFPGAN/basicsr/facexlib,
así que también baja bastante el tamaño de la instalación, no solo el
de los pesos.

Los pesos **no están en este repo** — se descargan de Hugging Face Hub la
primera vez que corre cada función, y se guardan en `/tmp` (por eso el primer
request tarda más, igual que en tu sistema anterior).

⚠️ **Parte sin probar en vivo:** la integración de `GPEN-BFR-512.onnx` en
`lib/pipeline.py` (funciones `_preprocess_gpen` / `_postprocess_gpen` /
`_enhance_face`) sigue la convención más común de estos modelos, pero no
la pude ejecutar aquí. Si el resultado sale con colores raros, muy oscuro
o muy claro, ese es el primer lugar a revisar/ajustar.

## ⚠️ Límite real de Vercel — leer antes de desplegar

Este código es funcional como referencia, pero **es muy probable que no
corra de forma confiable en el plan Hobby de Vercel**, y sigue siendo
justo en el plan Pro:

- **/tmp tiene límite de espacio** (512 MB en Hobby, hasta 3 GB en Fluid
  Compute de Pro). Con la versión ligera (~425 MB de pesos) **ya cabe justo**
  dentro del límite de Hobby, algo que con los modelos originales (1.2 GB)
  era imposible.
- **Tiempo de ejecución**: descargar ~425 MB en frío es bastante más rápido
  que 1.2 GB, pero sigue sumando al primer request de cada instancia fría.
- **CPU vs GPU**: sin GPU, la inferencia de estos 3 modelos en cadena puede
  tardar varios segundos por imagen — ya no es "imposible" como con los
  modelos pesados, pero sigue siendo más lento que con GPU.

**Recomendación real:** usa este repo tal cual para tener el código en
GitHub y versionado, pero para que funcione de forma confiable, corre la
inferencia (detector + swapper + enhancer) en un host con GPU:
- **Replicate** o **fal.ai** (ya tienen versiones alojadas de este mismo
  pipeline, pagas por uso, sin gestionar modelos tú mismo)
- **Modal** o **RunPod** (GPU por minuto, tú controlas el código)
- **Hugging Face Inference Endpoints**

Y deja que Vercel solo sirva el frontend y haga de proxy hacia esa API.
El código en `lib/pipeline.py` está separado justamente para poder moverlo
a cualquiera de esos hosts sin reescribirlo.

## Estructura

```
frontend/index.html     — interfaz (sube 2 fotos, llama a /api)
api/index.py            — app Flask única: /api/warmup y /api/swap-upload
lib/pipeline.py         — lógica de detección + swap + mejora
pyproject.toml          — dependencias + entrypoint que exige Vercel
vercel.json             — configuración de rutas y límites de función
```

**Nota sobre el cambio de `requirements.txt` a `pyproject.toml`:** Vercel
dejó de auto-detectar varios archivos sueltos en `api/` como funciones
independientes (así estaba armado al principio) — ahora exige un único
punto de entrada declarado en `pyproject.toml` vía `[tool.vercel]
entrypoint`. Por eso ambos endpoints ahora viven en una sola app Flask
(`api/index.py`) en vez de dos archivos separados.

## Subir a GitHub

1. Crea un repositorio nuevo vacío en GitHub (sin README, sin .gitignore).
2. En tu computadora, dentro de esta carpeta:
   ```bash
   git init
   git add .
   git commit -m "Proyecto face swap unificado"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   git push -u origin main
   ```
3. Conecta ese repo a un proyecto nuevo de Vercel (import desde GitHub, o
   pídeme que lo conecte yo una vez esté en GitHub).

No hace falta Git LFS: no estamos subiendo ningún archivo de modelo, solo
código — el repo completo pesa unos KB.
