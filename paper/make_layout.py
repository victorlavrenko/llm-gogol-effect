from pathlib import Path
import re

src_path = Path(__file__).with_name('main.tex')
out_path = Path(__file__).with_name('main_layout.tex')
s = src_path.read_text(encoding='utf-8')

# Typography and page-breaking policy: full-height columns, no 1- or 2-line
# paragraph fragments at column/page breaks, and enough room under headings.
s = s.replace(
    r'\usepackage{float}' + '\n',
    r'\usepackage{float}' + '\n' + r'\usepackage{needspace}' + '\n' + r'\usepackage{flushend}' + '\n'
)
s = s.replace(r'\raggedbottom', r'''\flushbottom
\emergencystretch=1.2em
\tolerance=1200
\pretolerance=700
\widowpenalties 3 10000 10000 0
\clubpenalties 3 10000 10000 0
\displaywidowpenalty=10000''')

# Keep the title full width, but place the abstract in the left column.
pat = re.compile(
    r'\\begin\{document\}\n'
    r'\\twocolumn\[\n'
    r'\\begin\{@twocolumnfalse\}\n'
    r'\\maketitle\n'
    r'\\vspace\{-1\.2em\}\n'
    r'\\begin\{abstract\}\n'
    r'(.*?)\n'
    r'\\end\{abstract\}\n'
    r'\\vspace\{0\.8em\}\n'
    r'\\end\{@twocolumnfalse\}\n'
    r'\]',
    re.S,
)
m = pat.search(s)
if not m:
    raise SystemExit('Could not locate title/abstract block')
abstract = m.group(1)
replacement = r'''\begin{document}
\twocolumn[
\begin{@twocolumnfalse}
\maketitle
\vspace{-1.0em}
\end{@twocolumnfalse}
]
\begin{abstract}
''' + abstract + r'''
\end{abstract}
\vspace{0.4em}'''
s = s[:m.start()] + replacement + s[m.end():]

# Keep headings with roughly three body lines when they begin near a break.
# Using a moderate reserve avoids creating visibly empty column bottoms.
s = re.sub(r'(?m)^\\section\{', lambda m: r'\Needspace{4.4\baselineskip}' + '\n' + m.group(0), s)
s = re.sub(r'(?m)^\\subsection\{', lambda m: r'\Needspace{3.8\baselineskip}' + '\n' + m.group(0), s)

# arXiv disclosure of significant generative-AI assistance.
disclosure = r'''\section*{AI Assistance Disclosure}
Generative AI tools assisted with manuscript drafting and editing, experimental workflow organization, code development and debugging, and discussion of experimental design and statistical analysis. The author made all final methodological decisions, executed and inspected the experiments, verified the reported results and citations, determined the interpretations and conclusions, and takes full responsibility for the work.

'''
if 'AI Assistance Disclosure' not in s:
    marker = r'\section*{Ethics Statement}'
    if marker not in s:
        raise SystemExit('Could not locate Ethics Statement')
    s = s.replace(marker, disclosure + marker, 1)

out_path.write_text(s, encoding='utf-8')
print(out_path)
