PDFLATEX := pdflatex
BIBTEX := bibtex
clean:
	rm -f *.aux
	rm -f *.bbl
	rm -f *.blg
	rm -f *.out
	rm -f *.log

%:
	make clean
	$(PDFLATEX) $@.tex
	$(PDFLATEX) $@.tex
