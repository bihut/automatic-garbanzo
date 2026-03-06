import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Usando dispositivo: {device}, dtype: {dtype}")

    # Cargar imagen base del personaje
    image_path = Path(character["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(
            f"No se encontró la imagen del personaje en '{image_path}'. "
            "Coloca ahí la imagen generada por Gemini (o ajusta 'image_path' en el fichero de pipeline)."
        )
    print(f"Cargando imagen base desde {image_path}")
    base_image = load_image(str(image_path))

    # Configuración del modelo: Stable Video Diffusion (image-to-video)
    model_type = model_cfg.get("type", "svd_img2vid")
    if model_type != "svd_img2vid":
        raise ValueError(
            f"Tipo de modelo no soportado: {model_type}. "
            "Actualmente sólo se ha implementado 'svd_img2vid' (Stable Video Diffusion)."
        )

    pretrained_name = model_cfg["pretrained_name"]
    num_frames = model_cfg.get("num_frames", 25)
    fps = model_cfg.get("fps", 7)
    decode_chunk_size = model_cfg.get("decode_chunk_size", 4)
    motion_bucket_id = model_cfg.get("motion_bucket_id", 127)
    noise_aug_strength = model_cfg.get("noise_aug_strength", 0.02)
    width = model_cfg.get("width", 1024)
    height = model_cfg.get("height", 576)

    print(f"Cargando pipeline StableVideoDiffusion: {pretrained_name}")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_name,
        torch_dtype=dtype,
        variant="fp16",
    )

    # Opciones de memoria para GPUs (3090 / V100)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        if hasattr(pipe.unet, "enable_forward_chunking"):
            pipe.unet.enable_forward_chunking()

    seed = character.get("seed", 1234)
    generator = torch.Generator(device=device).manual_seed(seed)

    for clip in clips:
        name = clip["name"]
        prompt = clip.get("prompt", "")
        print(f"Generando clip '{name}' (prompt usado solo como referencia de escena)...")

        # SVD es puramente image-to-video: animamos la imagen base redimensionada
        image_resized = base_image.resize((width, height))

        video = pipe(
            image=image_resized,
            num_frames=num_frames,
            decode_chunk_size=decode_chunk_size,
            generator=generator,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
        )

        frames = video.frames[0]  # lista de PIL Images

        out_path = clips_dir / f"{name}.mp4"
        export_to_video(frames, out_path, fps=fps)
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

