# setup.py
from packaging.requirements import Requirement
from pathlib import Path
from setuptools import setup, find_packages
import sys
from typing import List, Tuple, Optional, Union, Any, Dict

def load_requirements(
    path: Optional[str] = "requirements.txt",
) -> List[str]:
    
    def load_line(path: str):
        rows = list()
        with open(path, "r", encoding="utf-8") as fp:
            for row in fp:
                # TODO: do something if conflict occurs
                rows.append(row)
        return rows

    output = load_line(path)
    return output


setup(
    name="charxiv",
    version="0.0.0",
    description="CharXiv (https://github.com/princeton-nlp/CharXiv)",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="",
    package_dir={"charxiv": "src"},
    packages=[
        "charxiv",
    ],
    python_requires=">=3.10",
    # install_requires=load_requirements("requirements.txt"),
    extras_require={
        # "dev": load_requirements("requirements-dev.txt") if Path("requirements-dev.txt").exists() else [],
    },
    include_package_data=True,
)
