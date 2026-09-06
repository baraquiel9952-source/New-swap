"""
Pipeline de face swap: detección + intercambio + mejora facial.

VERSIÓN LIGERA — usa variantes más chicas de los mismos modelos:

| Etapa        | Modelo original      | Peso   | Modelo ligero            | Peso   |
|--------------|-----------------------|--------|---------------------------|--------|
| Detección    | buffalo_l             | 326 MB | buffalo_sc                | 16 MB  |
| Swap         | inswapper_128 (FP32)  | 554 MB | inswapper_128_fp16        | 125 MB |
| Restauración | GFPGANv1.4 (PyTorch)  | 349 MB | GPEN-BFR-512 (ONNX)       | 284 MB |

Total: ~1.2 GB -> ~425 MB de pesos, y de paso se elimina la dependencia
de PyTorch/GFPGAN/basicsr/facexlib, porque GPEN-BFR-512 corre solo con
onnxruntime igual que el detector y el swapper. Eso reduce bastante el
tamaño de la instalación además del tamaño de los pesos.

Nota honesta: la integración de GPEN-BFR-512 (normalización de entrada/
salida del modelo) está armada según el formato más común publicado en
proyectos que lo usan (ReActor, facexlib), pero no la pude probar en
vivo. Si el resultado sale con colores raros o muy oscuro/claro, el
primer sospechoso es la normalización en `_preprocess_gpen` /
`_postprocess_gpen` -- son los puntos marcados abajo para ajustar.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

MODEL_DIR = "/tmp/models"
os.makedirs(MODEL_DIR, exist_ok=True)

_face_analyser = None
_swapper = None
_enhancer_session = None


def get_face_analyser():
    """Detección + landmarks + embedding (InsightFace buffalo_sc, versión ligera)."""
    global _face_analyser
    if _face_analyser is None:
        from insightface.app import FaceAnalysis
        _face_analyser = FaceAnalysis(
            name="buffalo_sc",              # <- antes: buffalo_l (326MB) -> ahora 16MB
            root=MODEL_DIR,
            providers=["CPUExecutionProvider"],
        )
        _face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    return _face_analyser


def get_swapper():
    """Modelo generativo de intercambio de rostro (inswapper_128, versión FP16)."""
    global _swapper
    if _swapper is None:
        import insightface
        model_path = hf_hub_download(
            repo_id="latark/MorphStream",
            filename="inswapper_128_fp16.onnx",   # <- antes: inswapper_128.onnx (554MB) -> ahora ~125MB
            cache_dir=MODEL_DIR,
        )
        _swapper = insightface.model_zoo.get_model(
            model_path, providers=["CPUExecutionProvider"]
        )
    return _swapper


def get_enhancer_session():
    """Restauración/nitidez facial (GPEN-BFR-512, ONNX puro -- sin PyTorch)."""
    global _enhancer_session
    if _enhancer_session is None:
        model_path = hf_hub_download(
            repo_id="fofr/comfyui",
            filename="facerestore_models/GPEN-BFR-512.onnx",
            cache_dir=MODEL_DIR,
        )
        _enhancer_session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    return _enhancer_session


def _decode(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_jpeg(img, quality=92):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("No se pudo codificar el resultado como JPEG")
    return buf.tobytes()


def _preprocess_gpen(face_bgr_512):
    """BGR uint8 512x512 -> tensor NCHW float32 en rango [-1, 1] (convención GPEN/ReActor)."""
    img = cv2.cvtColor(face_bgr_512, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # NCHW
    return img


def _postprocess_gpen(output_tensor):
    """Tensor de salida del modelo -> BGR uint8 512x512."""
    out = output_tensor[0]
    out = np.transpose(out, (1, 2, 0))          # CHW -> HWC
    out = (out * 0.5 + 0.5) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _enhance_face(full_img_bgr, analyser):
    """
    Recorta y alinea el rostro del resultado del swap, lo pasa por GPEN,
    y lo vuelve a pegar en la imagen completa usando la matriz inversa
    (mismo patrón que usa GFPGANer/facexlib internamente).
    """
    from insightface.utils import face_align

    faces = analyser.get(full_img_bgr)
    if not faces:
        return full_img_bgr  # no se detectó cara para mejorar; se deja el resultado del swap tal cual

    face = faces[0]
    aligned, M = face_align.norm_crop2(full_img_bgr, face.kps, image_size=512)

    session = get_enhancer_session()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_tensor = _preprocess_gpen(aligned)
    output = session.run([output_name], {input_name: input_tensor})[0]
    restored_face = _postprocess_gpen(output)

    # deshacer la alineación y pegar de vuelta sobre la imagen completa
    M_inv = cv2.invertAffineTransform(M)
    h, w = full_img_bgr.shape[:2]
    warped = cv2.warpAffine(restored_face, M_inv, (w, h), borderMode=cv2.BORDER_REPLICATE)

    mask = np.ones((512, 512), dtype=np.float32)
    mask = cv2.warpAffine(mask, M_inv, (w, h))
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    mask = cv2.GaussianBlur(mask, (15, 15), 0)[..., np.newaxis]

    blended = (warped.astype(np.float32) * mask + full_img_bgr.astype(np.float32) * (1 - mask))
    return blended.astype(np.uint8)


def swap_and_enhance(source_bytes: bytes, target_bytes: bytes) -> bytes:
    """Ejecuta las 3 etapas (versión ligera) y devuelve la imagen resultado en JPEG."""
    analyser = get_face_analyser()
    swapper = get_swapper()

    source_img = _decode(source_bytes)
    target_img = _decode(target_bytes)
    if source_img is None or target_img is None:
        raise ValueError("No se pudo leer una de las dos imágenes")

    source_faces = analyser.get(source_img)
    target_faces = analyser.get(target_img)
    if not source_faces:
        raise ValueError("No se detectó ningún rostro en la foto de origen")
    if not target_faces:
        raise ValueError("No se detectó ningún rostro en la foto de destino")

    source_face = source_faces[0]
    target_face = target_faces[0]

    result = swapper.get(target_img.copy(), target_face, source_face, paste_back=True)

    try:
        enhanced = _enhance_face(result, analyser)
    except Exception:
        # si la mejora falla por lo que sea, se entrega igual el resultado del swap sin mejorar
        enhanced = result

    return _encode_jpeg(enhanced)


def warmup_status():
    """Fuerza la carga de los 3 modelos y reporta su estado (para /api/warmup)."""
    status = {}
    for name, loader in (
        ("detector", get_face_analyser),
        ("swapper", get_swapper),
        ("enhancer", get_enhancer_session),
    ):
        try:
            loader()
            status[name] = 200
        except Exception as exc:  # noqa: BLE001
            status[name] = f"error: {exc}"
    return status
