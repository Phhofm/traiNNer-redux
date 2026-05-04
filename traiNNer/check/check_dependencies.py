import importlib.metadata
import sys
import tomllib

from packaging.version import InvalidVersion, Version


def get_min_versions_from_pyproject() -> dict[str, str | None]:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    try:
        dependencies = pyproject["project"]["dependencies"]
    except KeyError as err:
        raise RuntimeError("No dependencies found in pyproject.toml") from err

    min_versions = {}
    import re
    for dep in dependencies:
        # Match the package name at the beginning, stopping at the first version operator or space
        match = re.match(r"^([a-zA-Z0-9\-_]+)", dep)
        if match:
            package = match.group(1).strip()
            if ">=" in dep:
                # Still try to extract min version if present for >= check
                version_match = re.search(r">=\s*([0-9a-zA-Z\.\-]+)", dep)
                min_versions[package] = version_match.group(1).strip() if version_match else None
            else:
                min_versions[package] = None

    return min_versions


def check_dependencies() -> None:
    min_versions = get_min_versions_from_pyproject()

    if sys.platform == "win32":
        cmd = "./install.bat"
    else:
        cmd = "./install.sh"

    for package, min_version in min_versions.items():
        try:
            installed_version = importlib.metadata.version(package)
        except Exception:
            raise RuntimeError(
                f"{package} is not installed. Please run this command to install: {cmd}"
            ) from None

        if min_version:
            try:
                if Version(installed_version) < Version(min_version):
                    raise RuntimeError(
                        f"{package} version {installed_version} is lower than the required version {min_version}. Please run this command to update dependencies: {cmd}"
                    )
            except InvalidVersion:
                raise RuntimeError(
                    f"Invalid version format for {package}: {installed_version}"
                ) from None

        # print(f"{package}: {installed_version} (OK)")
