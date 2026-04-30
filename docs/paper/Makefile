# Makefile for HIL-SERL Paper
#
# Compile:  make
# Clean:    make clean
# View:     make view (macOS Preview)

PAPER = paper

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex figures/*.pdf
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex

view: $(PAPER).pdf
	open $(PAPER).pdf

clean:
	latexmk -c
	rm -f $(PAPER).bbl $(PAPER).run.xml $(PAPER).synctex.gz

distclean: clean
	rm -f $(PAPER).pdf

.PHONY: all view clean distclean
