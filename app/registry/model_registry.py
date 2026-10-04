import json

import shutil

from pathlib import Path

from datetime import datetime


REGISTRY_FILE = Path(
    "models/registry.json"
)


PRODUCTION_DIR = Path(
    "models/production"
)


# ============================================================
# LOAD REGISTRY
# ============================================================

def load_registry():

    if not REGISTRY_FILE.exists():

        return {

            "current_version":
                None,

            "versions":
                [],
        }


    with open(

        REGISTRY_FILE,

        "r",

        encoding="utf-8",

    ) as file:

        return json.load(
            file
        )


# ============================================================
# SAVE REGISTRY
# ============================================================

def save_registry(
    registry,
):

    REGISTRY_FILE.parent.mkdir(

        parents=True,

        exist_ok=True,
    )


    temp_file = (
        REGISTRY_FILE.with_suffix(
            ".tmp"
        )
    )


    with open(

        temp_file,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            registry,

            file,

            indent=2,
        )


    temp_file.replace(
        REGISTRY_FILE
    )


# ============================================================
# PROMOTE MODEL
# ============================================================

def promote_model(

    training_model_dir: str,

    metrics: dict,
):

    timestamp = (

        datetime.utcnow()

        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )


    version = (
        f"v{timestamp}"
    )


    production_dir = (

        PRODUCTION_DIR
        /
        version
    )


    PRODUCTION_DIR.mkdir(

        parents=True,

        exist_ok=True,
    )


    shutil.copytree(

        training_model_dir,

        production_dir,
    )


    registry = (
        load_registry()
    )


    registry[
        "current_version"
    ] = version


    registry.setdefault(

        "versions",

        [],
    )


    registry[
        "versions"
    ].append({

        "version":
            version,

        "path":
            str(
                production_dir
            ),

        "metrics":
            metrics,

        "created_at":
            datetime.utcnow()
            .isoformat(),
    })


    save_registry(
        registry
    )


    return version


# ============================================================
# GET CURRENT PRODUCTION
# ============================================================

def get_production_model():

    registry = (
        load_registry()
    )


    current_version = (
        registry.get(
            "current_version"
        )
    )


    if not current_version:

        return None


    for version in registry.get(

        "versions",

        [],
    ):

        if (

            version.get(
                "version"
            )

            ==

            current_version

        ):

            return version


    return None