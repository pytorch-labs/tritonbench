import subprocess
from setuptools import setup

def get_git_commit_hash(length=8):
    try:
        cmd = ['git', 'rev-parse', f'--short={length}', 'HEAD']
        return "+git{}".format(subprocess.check_output(cmd).strip().decode('utf-8'))
    except Exception:
        return ""

setup(
    name="tritonbench",
    version="0.0.1" + get_git_commit_hash(),
    author="Xu Zhao",
    author_email="xzhao9@meta.com",
    description="A benchmark suite for OpenAI Triton and TorchInductor",
    long_description="",
)