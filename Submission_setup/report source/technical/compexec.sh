pdflatex -shell-escape main.tex 
bibtex main
pdflatex -shell-escape main.tex 

rm main.aux
rm main.bbl
rm main.bcf
rm main.blg
rm main.log
rm main.out
rm main.run.xml
rm main.toc