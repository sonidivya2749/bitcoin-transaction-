import magic
from pathlib import Path

ALLOWED_FORMATS = {".csv", ".json", ".xml"}

MAX_FILE_SIZE_MB = 500
MAX_ROWS = 500_000

MAX_JSON_DEPTH = 50
MAX_XML_DEPTH = 50

MAX_PATH_LENGTH = 260

ALLOWED_MIME_TYPES = {
    ".csv": {"text/plain", "text/csv"},
    ".json": {"text/plain", "application/json"},
    ".xml": {"text/plain", "text/xml", "application/xml"},
}


def validate_file_content(filepath):
    # Verify detected content is compatible with the file extension.
    suffix = filepath.suffix.lower()
    mime_type = magic.from_file(str(filepath), mime=True)

    if mime_type not in ALLOWED_MIME_TYPES[suffix]:
        raise ValueError(
            f"File content does not match {suffix} format: {mime_type}"
        )

    return True


def validate_file_security(filepath):
    # Validate file type, path, size, and basic filesystem safety.
    path = Path(filepath)

    if len(str(path)) > MAX_PATH_LENGTH:
        raise ValueError("Input path is too long")

    if path.is_symlink():
        raise ValueError("Symbolic links are not allowed")

    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Input file not found: {path}"
        ) from exc

    if not resolved.is_file():
        raise ValueError(f"Input path is not a file: {resolved}")

    if resolved.suffix.lower() not in ALLOWED_FORMATS:
        raise ValueError(
            f"Unsupported file format '{resolved.suffix}'. "
            f"Allowed formats: CSV, JSON, XML"
        )

    if resolved.stat().st_size == 0:
        raise ValueError("Input file is empty")

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    if resolved.stat().st_size > max_bytes:
        raise ValueError(
            f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB"
        )

    validate_file_content(resolved)

    return resolved


def validate_row_limit(row_count, dataset_name):
    # Reject datasets exceeding the configured row limit.
    if row_count > MAX_ROWS:
        raise ValueError(
            f"{dataset_name}: {row_count:,} rows exceed the "
            f"maximum allowed {MAX_ROWS:,} rows"
        )

    return True


def validate_depth(depth, maximum, format_name):
    # Prevent excessively nested structured input.
    if depth > maximum:
        raise ValueError(
            f"{format_name}: nesting depth exceeds maximum of {maximum}"
        )

    return True


def validate_json_depth(data, maximum=MAX_JSON_DEPTH):
    # Validate maximum nesting depth of parsed JSON data.
    stack = [(data, 1)]

    while stack:
        value, depth = stack.pop()

        validate_depth(depth, maximum, "JSON")

        if isinstance(value, dict):
            for child in value.values():
                stack.append((child, depth + 1))

        elif isinstance(value, list):
            for child in value:
                stack.append((child, depth + 1))

    return True


def validate_xml_depth(root, maximum=MAX_XML_DEPTH):
    # Validate maximum element nesting depth of parsed XML.
    stack = [(root, 1)]

    while stack:
        element, depth = stack.pop()

        validate_depth(depth, maximum, "XML")

        for child in element:
            stack.append((child, depth + 1))

    return True