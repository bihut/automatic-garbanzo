import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import requests


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_timestamped_output(base_dir: Path, run_name: Optional[str] = None) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    if run_name:
        run_dir = base_dir / f"{ts}_{run_name}"
    else:
        run_dir = base_dir / ts
    (run_dir / "clips").mkdir(parents=True, exist_ok=True)
    return run_dir


def prepare_image_payload(image_path: str) -> str:
    """Devuelve una cadena para image_url: si es URL http, se usa tal cual;
    si es ruta local, se convierte a data URL base64."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró la imagen del personaje en '{path}'. "
            "Coloca ahí la imagen generada por Gemini o ajusta 'image_path' en el fichero de pipeline."
        )

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/png"

    with open(path, "rb") as f:
        data = f.read()

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def get_api_key(model_cfg: Dict[str, Any]) -> str:
    env_name = model_cfg.get("api_key_env", "AIMLAPI_API_KEY")
    api_key = os.getenv(env_name)
    if not api_key:
        raise RuntimeError(
            f"No se encontró la API key. Define la variable de entorno '{env_name}' "
            "con tu clave de AIMLAPI (Hailuo)."
        )
    return api_key


def hailuo_create_task(
    api_key: str,
    base_url: str,
    model_name: str,
    prompt: str,
    image_url: str,
    duration: int,
    resolution: str,
    enhance_prompt: bool,
) -> Dict[str, Any]:
    url = f"{base_url}/video/generations"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "image_url": image_url,
        "duration": duration,
        "resolution": resolution,
        "enhance_prompt": enhance_prompt,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Error Hailuo create_task {resp.status_code}: {resp.text}")
    return resp.json()


def hailuo_poll_task(
    api_key: str,
    base_url: str,
    generation_id: str,
    poll_interval: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    url = f"{base_url}/video/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    start = time.time()

    while True:
        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Timeout al esperar el vídeo de Hailuo (id={generation_id})."
            )

        resp = requests.get(
            url, params={"generation_id": generation_id}, headers=headers, timeout=60
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Error Hailuo poll_task {resp.status_code}: {resp.text}")

        data = resp.json()
        status = data.get("status")
        print(f"Estado de la tarea {generation_id}: {status}")

        if status in ("queued", "waiting", "generating"):
            time.sleep(poll_interval)
            continue

        if status == "completed":
            return data

        # status == "error" u otros
        raise RuntimeError(f"Tarea Hailuo en estado de error: {data}")


def download_video(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def main(config_path: str = "pipelines/pipeline1/nina_alegre_demo_v1.json") -> None:
    cfg = load_config(config_path)

    character = cfg["character"]
    model_cfg = cfg["model"]
    clips = cfg.get("clips", [])
    output_cfg = cfg.get("output", {})

    base_output_dir = Path(output_cfg.get("base_dir", "outputs")).expanduser()
    run_name = cfg.get("run_name")
    run_dir = create_timestamped_output(base_output_dir, run_name)
    clips_dir = run_dir / "clips"

    pipeline_id = cfg.get("id", "unknown_id")
    evaluation = cfg.get("evaluation", 0)
    print(f"ID del pipeline: {pipeline_id} | Evaluación: {evaluation}")
    print(f"Directorio de salida de esta ejecución: {run_dir}")

    # Configuración del modelo Hailuo (via AIMLAPI)
    model_type = model_cfg.get("type", "hailuo_i2v")
    if model_type != "hailuo_i2v":
        raise ValueError(
            f"Tipo de modelo no soportado: {model_type}. "
            "Actualmente sólo se ha implementado 'hailuo_i2v' (Hailuo 2.3 vía AIMLAPI)."
        )

    api_key = get_api_key(model_cfg)
    base_url = model_cfg.get("base_url", "https://api.aimlapi.com/v2")
    hailuo_model = model_cfg.get("hailuo_model", "minimax/hailuo-2.3")
    duration = int(model_cfg.get("duration", 6))
    resolution = model_cfg.get("resolution", "768P")
    enhance_prompt = bool(model_cfg.get("enhance_prompt", True))
    poll_interval = int(model_cfg.get("poll_interval", 15))
    timeout_seconds = int(model_cfg.get("timeout_seconds", 900))

    image_path = character["image_path"]
    print(f"Preparando imagen base desde {image_path}")
    image_url_payload = prepare_image_payload(image_path)

    for clip in clips:
        name = clip["name"]
        prompt = clip.get("prompt", "")

        print(f"Creando tarea de vídeo para clip '{name}' en Hailuo...")
        task_resp = hailuo_create_task(
            api_key=api_key,
            base_url=base_url,
            model_name=hailuo_model,
            prompt=prompt,
            image_url=image_url_payload,
            duration=duration,
            resolution=resolution,
            enhance_prompt=enhance_prompt,
        )

        gen_id = task_resp.get("generation_id") or task_resp.get("id")
        if not gen_id:
            raise RuntimeError(f"No se recibió generation_id/id en la respuesta: {task_resp}")

        print(f"Tarea creada con id: {gen_id}. Esperando resultado...")
        result = hailuo_poll_task(
            api_key=api_key,
            base_url=base_url,
            generation_id=gen_id,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )

        video_info = result.get("video") or {}
        video_url = video_info.get("url")
        if not video_url:
            raise RuntimeError(f"No se encontró URL de vídeo en la respuesta final: {result}")

        out_path = clips_dir / f"{name}.mp4"
        print(f"Descargando vídeo de '{video_url}' a '{out_path}'")
        download_video(video_url, out_path)
        print(f"Guardado clip: {out_path}")

    print("Todos los clips se han generado correctamente.")


if __name__ == "__main__":
    # Permite llamar:
    #   python image_to_video_pipeline.py
    #   python image_to_video_pipeline.py pipelines/otro_pipeline.json
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()

