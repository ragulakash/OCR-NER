import json

from pathlib import Path
from datetime import datetime


from app.discovery.candidates import (
    load_candidates,
    save_candidates,
    get_candidate,
    update_type_prediction,
)

from app.discovery.learning_queue import (
    load_queue,
    save_queue,
    WORKER_LOCK,
)

from app.learning.type_classifier import (
    get_type_classifier,
)

from app.learning.gemini_validator import (
    get_gemini_validator,
)

from app.learning.replay_buffer import (
    add_replay_example,
)


# ============================================================
# CONFIGURATION
# ============================================================

REPLAY_MIN_EXAMPLES = 10

DEMO_CONFIRMATION_CONFIDENCE = 0.60

MIN_OCCURRENCES = 3

MIN_CONSISTENCY = 0.70


# ============================================================
# LOG
# ============================================================

def log(message):

    print(
        f"[TYPE LEARNING] {message}"
    )


# ============================================================
# CONTEXT HELPERS
# ============================================================

def get_contexts(candidate):

    contexts = (
        candidate.get(
            "contexts",
            [],
        )
    )

    if not isinstance(
        contexts,
        list,
    ):

        return []

    return contexts


def get_context_text(item):

    if isinstance(
        item,
        dict,
    ):

        return str(
            item.get(
                "context",
                item.get(
                    "text",
                    "",
                ),
            )
        )

    return str(item)


# ============================================================
# CONFIRMATION
# ============================================================

def force_demo_confirmation(
    candidate,
):

    if not candidate:

        return candidate

    occurrences = int(
        candidate.get(
            "occurrences",
            0,
        )
    )

    confidence = float(
        candidate.get(
            "type_confidence",
            0.0,
        )
        or 0.0
    )

    consistency = float(
        candidate.get(
            "type_consistency",
            0.0,
        )
        or 0.0
    )

    entity_type = candidate.get(
        "type"
    )

    if not entity_type:

        return candidate

    if (
        occurrences >= MIN_OCCURRENCES
        and confidence
            >= DEMO_CONFIRMATION_CONFIDENCE
        and consistency
            >= MIN_CONSISTENCY
    ):

        candidates = load_candidates()

        key = candidate.get(
            "normalized"
        )

        if key in candidates:

            candidates[key][
                "status"
            ] = "CONFIRMED"

            candidates[key][
                "type"
            ] = entity_type

            candidates[key][
                "updated_at"
            ] = datetime.utcnow().isoformat()

            save_candidates(
                candidates
            )

            candidate = candidates[key]

            log(
                f"CONFIRMED: "
                f"{candidate.get('text')} "
                f"→ "
                f"{entity_type} "
                f"(confidence="
                f"{confidence:.4f})"
            )

    return candidate


# ============================================================
# REPLAY CREATION
# ============================================================

def create_replay_for_candidate(
    candidate,
):

    if not candidate:

        return 0

    if candidate.get(
        "status"
    ) != "CONFIRMED":

        return 0

    entity_text = str(
        candidate.get(
            "text",
            "",
        )
    ).strip()

    entity_type = str(
        candidate.get(
            "type",
            "",
        )
    ).upper().strip()

    if (
        not entity_text
        or not entity_type
    ):

        return 0

    added = 0

    for item in get_contexts(
        candidate
    ):

        context = get_context_text(
            item
        ).strip()

        if not context:

            continue

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Candidate start/end are document-level offsets.
        # Context is a smaller substring.
        #
        # Therefore DO NOT reuse the old offsets.
        #
        # Find the entity inside the context.
        # ----------------------------------------------------

        start = context.find(
            entity_text
        )

        if start < 0:

            start = context.lower().find(
                entity_text.lower()
            )

        if start < 0:

            continue

        end = (
            start
            + len(entity_text)
        )

        added += add_replay_example(
            text=context,
            label=entity_type,
            entity_text=entity_text,
            start=start,
            end=end,
        )

    return added


# ============================================================
# TRAINING TRIGGER
# ============================================================

def trigger_ner_training():

    try:

        from app.learning.replay_buffer import (
            load_replay
        )

        replay = load_replay()

        log(
            f"Replay size: "
            f"{len(replay)}"
        )

        if len(replay) < REPLAY_MIN_EXAMPLES:

            log(
                "Replay threshold not reached."
            )

            return

        from app.workers.training_worker import (
            start_training_process
        )

        result = (
            start_training_process()
        )

        log(
            f"NER training trigger: "
            f"{result}"
        )

    except Exception as error:

        log(
            f"Training trigger failed: "
            f"{error}"
        )


# ============================================================
# PROCESS ENTITY
# ============================================================

def process_entity(
    entity_text,
):

    candidate = get_candidate(
        entity_text
    )

    if not candidate:

        log(
            f"No candidate found: "
            f"{entity_text}"
        )

        return "SKIPPED"

    # --------------------------------------------------------
    # FIRST TRY CONFIRMATION USING EXISTING EVIDENCE
    # --------------------------------------------------------

    candidate = (
        force_demo_confirmation(
            candidate
        )
    )

    if candidate.get(
        "status"
    ) == "CONFIRMED":

        create_replay_for_candidate(
            candidate
        )

        trigger_ner_training()

        return "CONFIRMED"

    # --------------------------------------------------------
    # TYPE CLASSIFIER & GEMINI VALIDATOR
    # --------------------------------------------------------

    classifier = get_type_classifier()
    gemini = get_gemini_validator()

    contexts = get_contexts(candidate)

    if not contexts:
        log(f"No contexts for {entity_text}")
        return "LEARNING"

    first_context = get_context_text(contexts[0]).strip()

    # --------------------------------------------------------
    # DISTILBERT TYPE PREDICTION
    # --------------------------------------------------------

    top_confidence = 0.0

    if classifier.is_available():
        for item in contexts:
            context = get_context_text(item).strip()
            if not context:
                continue

            log(f"Classifying with DistilBERT: {entity_text}")
            predictions = classifier.predict(entity_text, context)

            if not predictions:
                continue

            best = predictions[0]
            predicted_type = best.get("type")
            confidence = float(best.get("confidence", 0.0) or 0.0)

            if not predicted_type:
                continue

            top_confidence = max(top_confidence, confidence)

            log(f"Prediction: {entity_text} → {predicted_type} ({confidence:.4f})")

            candidate = update_type_prediction(
                text=entity_text,
                predicted_type=predicted_type,
                confidence=confidence,
            )

            if not candidate:
                continue

            candidate = force_demo_confirmation(candidate)
            if candidate.get("status") == "CONFIRMED":
                break

    # --------------------------------------------------------
    # OPTIONAL GEMINI TEACHER VALIDATION FOR LOW CONFIDENCE
    # --------------------------------------------------------

    GEMINI_CONFIDENCE_THRESHOLD = 0.75

    if (
        candidate.get("status") != "CONFIRMED"
        and top_confidence < GEMINI_CONFIDENCE_THRESHOLD
        and gemini.is_available()
    ):
        log(f"Low DistilBERT confidence ({top_confidence:.4f} < {GEMINI_CONFIDENCE_THRESHOLD}). Requesting Gemini teacher validation...")

        g_res = gemini.validate_candidate(
            entity_text=entity_text,
            context=first_context,
            proposed_type=candidate.get("proposed_type"),
        )

        if g_res and float(g_res.get("confidence", 0.0)) >= 0.70:
            gemini_type = g_res["type"]
            gemini_conf = float(g_res["confidence"])

            log(f"Gemini Teacher Confirmed: '{entity_text}' → {gemini_type} ({gemini_conf:.4f})")

            candidate = update_type_prediction(
                text=entity_text,
                predicted_type=gemini_type,
                confidence=gemini_conf,
            )

            # Direct confirmation via teacher evidence
            candidates = load_candidates()
            norm = candidate.get("normalized")
            if norm and norm in candidates:
                candidates[norm]["status"] = "CONFIRMED"
                candidates[norm]["type"] = gemini_type
                candidates[norm]["type_confidence"] = gemini_conf
                candidates[norm]["updated_at"] = datetime.utcnow().isoformat()
                save_candidates(candidates)
                candidate = candidates[norm]

    # --------------------------------------------------------
    # CONFIRMED
    # --------------------------------------------------------

    if candidate.get("status") == "CONFIRMED":
        added = create_replay_for_candidate(candidate)
        log(f"Replay examples added: {added}")
        trigger_ner_training()
        return "CONFIRMED"

    # --------------------------------------------------------
    # STILL LEARNING
    # --------------------------------------------------------

    log(f"Still learning: {entity_text}")
    return "LEARNING"


# ============================================================
# QUEUE
# ============================================================

def process_queue():

    queue = load_queue()

    if not queue:
        log("Queue is empty.")
        return

    classifier = get_type_classifier()
    gemini = get_gemini_validator()

    if not classifier.is_available() and not gemini.is_available():
        log("Neither Type Classifier nor Gemini is available.")
        return

    remaining = []

    for entity_text in queue:

        try:

            status = process_entity(
                entity_text
            )

            if status != "CONFIRMED":

                remaining.append(
                    entity_text
                )

        except Exception as error:

            log(
                f"Error processing "
                f"{entity_text}: "
                f"{error}"
            )

            remaining.append(
                entity_text
            )

    save_queue(
        remaining
    )

    log(
        f"Queue remaining: "
        f"{len(remaining)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        process_queue()

    finally:

        WORKER_LOCK.unlink(
            missing_ok=True
        )


if __name__ == "__main__":

    main()