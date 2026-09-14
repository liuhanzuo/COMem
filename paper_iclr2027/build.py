"""Build the manuscript with its numbered, clickable bibliography."""
from pathlib import Path
import shutil
import subprocess


def main():
    paper = Path(__file__).resolve().parent
    build = paper / 'build'
    build.mkdir(exist_ok=True)
    for source in [paper / 'refs.bib', *paper.glob('*.bst')]:
        shutil.copy2(source, build / source.name)
    command = ['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
               '-halt-on-error', '-output-directory=' + build.as_posix(), 'main.tex']
    for run in range(3):
        result = subprocess.run(command, cwd=paper, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (build / f'compile_{run + 1}.txt').write_bytes(result.stdout)
        if result.returncode:
            raise SystemExit(result.stdout.decode(errors='replace'))
        if run == 0:
            result = subprocess.run(['bibtex', 'main'], cwd=build,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (build / 'bibtex.txt').write_bytes(result.stdout)
            if result.returncode:
                raise SystemExit(result.stdout.decode(errors='replace'))
    log = (build / 'main.log').read_text(errors='replace')
    issues = [line for line in log.splitlines() if any(term in line for term in
              ('Overfull', 'undefined', 'multiply defined', 'Rerun to get'))]
    if issues:
        raise SystemExit('\n'.join(issues))
    print(build / 'main.pdf')


if __name__ == '__main__':
    main()
