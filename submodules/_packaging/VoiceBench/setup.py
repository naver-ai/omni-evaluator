# setup.py
from importlib.metadata import distributions
from packaging.requirements import Requirement
from pathlib import Path
import re
from setuptools import setup, find_packages
import sys
from typing import List, Tuple, Optional, Union, Any, Dict

installed_package_names = set([
    _distribution.metadata["Name"]
    # (_distribution.metadata["Name"], _distribution.version)
    for _distribution in distributions()
])

def load_requirements(
    path: Optional[str] = "requirements.txt",
) -> List[str]:
    
    def load_line(path: str):
        rows = list()
        with open(path, "r", encoding="utf-8") as fp:
            for row in fp:
                # TODO: do something if conflict occurs
                _package_name = row
                _match = re.search(r'[><=]', row)
                if _match:
                    _package_name = row[:_match.start()]
                if _package_name in installed_package_names:
                    print("!!!")
                    continue
                rows.append(row)
        return rows

    output = load_line(path)
    return output


setup(
    name="voice_bench",
    version="0.0.0",
    description="VoiceBench (https://github.com/MatthewCYM/VoiceBench)",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="",
    package_dir={"voice_bench": "src"},
    packages=["voice_bench"] + ["voice_bench." + _p for _p in find_packages(where="src")],
    python_requires=">=3.10",
    # install_requires=load_requirements("requirements.txt"),
    extras_require={
        # "dev": load_requirements("requirements-dev.txt") if Path("requirements-dev.txt").exists() else [],
    },
    include_package_data=True,
)
