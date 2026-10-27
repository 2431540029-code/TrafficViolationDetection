CARGO_CLASS_NAMES = frozenset({
    "bulky_cargo",
    "protrusion",
    "xe_cong_kenh",
})


def normalize_cargo_classes(names):
    if isinstance(names, dict):
        class_names = {
            int(class_id): str(name).strip().lower()
            for class_id, name in names.items()
        }
    else:
        class_names = {
            class_id: str(name).strip().lower()
            for class_id, name in enumerate(names)
        }

    cargo_class_ids = sorted(
        class_id
        for class_id, name in class_names.items()
        if name in CARGO_CLASS_NAMES
    )
    if not cargo_class_ids:
        raise ValueError(
            "Model cargo phai co mot trong cac class "
            f"{', '.join(sorted(CARGO_CLASS_NAMES))}; "
            f"cac class hien co: {class_names}"
        )

    normalized_names = {
        class_id: (
            "bulky_cargo"
            if class_id in cargo_class_ids
            else name
        )
        for class_id, name in class_names.items()
    }
    return cargo_class_ids, normalized_names
