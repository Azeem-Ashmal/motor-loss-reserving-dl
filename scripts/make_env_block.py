"""
Script to generate env_block.tex from requirements-lock.txt dynamically.
Ensures zero hardcoded version numbers exist in the thesis .tex file.
"""

import sys
import re
from pathlib import Path

def generate_env_block():
    repo_root = Path(__file__).resolve().parent.parent
    lockfile_path = repo_root / 'requirements-lock.txt'
    
    if not lockfile_path.exists():
        print(f"[ERROR] {lockfile_path} not found. Run pip freeze first.")
        sys.exit(1)
        
    with open(lockfile_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    pkgs = {}
    for line in content.splitlines():
        line = line.strip()
        if line.startswith('Python '):
            pkgs['Python'] = line.split('Python ')[1]
        elif '==' in line:
            name, ver = line.split('==', 1)
            pkgs[name.lower()] = ver
            
    required = ['Python', 'torch', 'scikit-learn', 'numpy', 'pandas']
    missing = [req for req in required if req.lower() not in [k.lower() for k in pkgs.keys()]]
    
    if missing:
        print(f"[ERROR] Missing required versions in lockfile: {missing}")
        sys.exit(1)

    py_ver = pkgs['Python']
    torch_ver = pkgs['torch']
    sklearn_ver = pkgs['scikit-learn']
    numpy_ver = pkgs['numpy']
    pandas_ver = pkgs['pandas']

    # Escape underscores for LaTeX
    def tex_escape(s):
        return s.replace('_', '\\_')

    latex_snippet = f"""\\begin{{itemize}}
  \\item \\textbf{{Python}}: \\texttt{{{tex_escape(py_ver)}}}
  \\item \\textbf{{PyTorch}}: \\texttt{{{tex_escape(torch_ver)}}}
  \\item \\textbf{{scikit-learn}}: \\texttt{{{tex_escape(sklearn_ver)}}}
  \\item \\textbf{{NumPy}}: \\texttt{{{tex_escape(numpy_ver)}}}
  \\item \\textbf{{pandas}}: \\texttt{{{tex_escape(pandas_ver)}}}
  \\item \\textbf{{Random Seed}}: \\texttt{{42}}
\\end{{itemize}}
"""

    env_block_repo = repo_root / 'env_block.tex'
    with open(env_block_repo, 'w', encoding='utf-8') as f:
        f.write(latex_snippet)
        
    # Also write to Deliverables folder for thesis compilation
    deliverables_dir = repo_root.parent / 'Deliverables'
    if deliverables_dir.exists():
        with open(deliverables_dir / 'env_block.tex', 'w', encoding='utf-8') as f:
            f.write(latex_snippet)
            
    print(f"[SUCCESS] Successfully generated env_block.tex at {env_block_repo}")

if __name__ == '__main__':
    generate_env_block()
