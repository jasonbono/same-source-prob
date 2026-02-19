#!/bin/bash
# Auto-compile LaTeX on file changes
fswatch -o main.tex | while read; do
    clear
    echo "Compiling..."
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
    echo "✓ Compiled at $(date +%H:%M:%S)"
done
