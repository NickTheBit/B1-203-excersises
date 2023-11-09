# Lecture 11 makefile, used to generate deliverables
lNum = 11
tempFolder = Lecture_$(lNum)_203
targetZip = $(tempFolder).zip

report.pdf: report.tex
	pdflatex report.tex
	rm report.aux
	rm report.log
	rm report.out

$(tempFolder).zip: report.pdf Lecture_$(lNum)_203.py
	mkdir $(tempFolder)
	cp report.pdf $(tempFolder)
	cp Lecture_$(lNum)_203.py $(tempFolder)
	zip -r $(tempFolder).zip $(tempFolder)
	rm -r $(tempFolder)

all: $(tempFolder).zip
report : report.pdf
