import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


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


def main(config_path: str = "pipelines/pipeline_ltx2/personajes_jugando_ltx2.json") -> None:
    """
    Ejecuta LTX-2 TI2VidTwoStagesPipeline usando la configuración de un fichero JSON
    compatible con tu estructura de pipelines.

    Requisitos previos (en el entorno donde lances este script):
    - Haber clonado e instalado el repo LTX-2, por ejemplo:
      git clone https://github.com/Lightricks/LTX-2.git
      cd LTX-2
      uv sync --frozen
      source .venv/bin/activate
      pip install -e packages/ltx-pipelines -e packages/ltx-core
    - Haber descargado los checkpoints y rutas que pondrás en el JSON.
    """
    # Importamos aquí para que sólo falle si realmente usas LTX-2
    try:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils import get_device
        from ltx_pipelines.utils.media_io import encode_video
    except ImportError as exc:
        raise RuntimeError(
            "No se pudieron importar los paquetes de LTX-2. "
            "Asegúrate de haber instalado el repo LTX-2 en este entorno "
            "(por ejemplo con 'pip install -e packages/ltx-core -e packages/ltx-pipelines')."
        ) from exc

    logging.getLogger().setLevel(logging.INFO)

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

    # Configuración básica del modelo LTX-2
    model_type = model_cfg.get("type", "ltx2_ti2vid_two_stages")
    if model_type != "ltx2_ti2vid_two_stages":
        raise ValueError(
            f"Tipo de modelo no soportado en este script: {model_type}. "
            "Usa 'ltx2_ti2vid_two_stages' en el bloque 'model' de tu JSON."
        )

    checkpoint_path = model_cfg["checkpoint_path"]
    spatial_upsampler_path = model_cfg["spatial_upsampler_path"]
    gemma_root = model_cfg["gemma_root"]

    # Distilled LoRA (recomendada por LTX-2 para el segundo stage)
    distilled_lora = []
    distilled_lora_path = model_cfg.get("distilled_lora_path")
    if distilled_lora_path:
        distilled_lora_strength = float(model_cfg.get("distilled_lora_strength", 0.6))
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                distilled_lora_path,
                distilled_lora_strength,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

    # LoRAs extra opcionales (estilo, cámara, etc.) — por defecto vacío
    loras: list[LoraPathStrengthAndSDOps] = []

    device = get_device()

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=loras,
        device=device,
        quantization=None,
    )

    # Parámetros generales de vídeo
    height = int(model_cfg.get("height", 512))
    width = int(model_cfg.get("width", 768))
    num_frames = int(model_cfg.get("num_frames", 121))
    frame_rate = float(model_cfg.get("frame_rate", 25.0))
    num_inference_steps = int(model_cfg.get("num_inference_steps", 40))

    # Guider params (puedes afinarlos en el JSON si quieres)
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=float(model_cfg.get("video_cfg_scale", 3.0)),
        stg_scale=float(model_cfg.get("video_stg_scale", 1.0)),
        rescale_scale=float(model_cfg.get("video_rescale_scale", 0.7)),
        modality_scale=float(model_cfg.get("video_modality_scale", 3.0)),
        skip_step=int(model_cfg.get("video_skip_step", 0)),
        stg_blocks=model_cfg.get("video_stg_blocks", [29]),
    )

    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=float(model_cfg.get("audio_cfg_scale", 7.0)),
        stg_scale=float(model_cfg.get("audio_stg_scale", 1.0)),
        rescale_scale=float(model_cfg.get("audio_rescale_scale", 0.7)),
        modality_scale=float(model_cfg.get("audio_modality_scale", 3.0)),
        skip_step=int(model_cfg.get("audio_skip_step", 0)),
        stg_blocks=model_cfg.get("audio_stg_blocks", [29]),
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    seed = int(character.get("seed", 1234))

    # NOTA: por simplicidad inicial NO usamos image conditioning aquí;
    # LTX-2 generará el vídeo solo a partir del prompt (muy recomendado
    # por ellos). Si más adelante quieres, podemos añadir conditioning
    # desde 'character.image_path'.
    images: list[Any] = []

    for clip in clips:
        name = clip["name"]
        prompt = clip["prompt"]
        negative_prompt = clip.get("negative_prompt", "")

        print(f"Generando clip '{name}' con LTX-2...")

        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=bool(model_cfg.get("enhance_prompt", False)),
        )

        out_path = clips_dir / f"{name}.mp4"
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=str(out_path),
            video_chunks_number=video_chunks_number,
        )
        print(f"Guardado clip: {out_path}")

    print("Todos los clips LTX-2 se han generado correctamente.")


if __name__ == "__main__":
    # Permite llamar:
    #   python ltx2_pipeline.py
    #   python ltx2_pipeline.py pipelines/lo_que_sea.json
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()

